"""实验运行公共层；不依赖 MuJoCo，也不导入控制器或预测模型。

最终参数和输入内容决定运行身份。历史结果及失败尝试始终保留；只有带有
相同身份且输出哈希完整的结果才可复用。socket 路径属于进程临时资源，不参与身份。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import time
import uuid

SCHEMA = "PIPELINE_EXPERIMENT/1"
CONTROL_PREFIXES = ("M2_", "NMPC6_", "N6_", "BALL_")
CONTROL_FLAGS = ("WB_CHASSIS", "WM_DEBUG", "WBO_DEBUG")
STEER_RATE_MODES = ("limited", "direction_unlimited", "feedforward_unlimited", "both_unlimited")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def clean_environment(overrides=None, inherited=None):
    """清理上次控制实验遗留开关；EXP_* 是现有独立实验的显式输入，保留并记录。"""
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(CONTROL_PREFIXES) or key in CONTROL_FLAGS:
            del env[key]
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.update({key: str(value) for key, value in (overrides or {}).items()})
    return env


def recorded_environment(env, socket=None):
    """只记录会改变算法/运行库的白名单变量，不保存整个进程环境或凭据。"""
    keys = {key for key in env if key.startswith(CONTROL_PREFIXES + ("EXP_",)) or key in CONTROL_FLAGS}
    keys.update(key for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHASHSEED")
                if key in env)
    return {key: ("{socket}" if socket is not None and env[key] == str(socket) else env[key])
            for key in sorted(keys)}


def file_resources(paths):
    """调用者提供明确文件/目录；目录逐文件哈希，不把路径或 mtime 当作内容身份。"""
    resources = {}
    for label, raw in sorted(paths.items()):
        path = Path(raw).resolve(strict=True)
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.is_file() and "__pycache__" not in item.parts:
                    resources[f"{label}/{item.relative_to(path)}"] = {
                        "path": str(item.resolve()), "sha256": digest(item)}
        elif path.is_file():
            resources[label] = {"path": str(path), "sha256": digest(path)}
        else:
            raise ValueError(f"not a regular resource: {path}")
    return resources


def team_resources(team, mode, *, binary=None):
    """包含生成场景、网格/配置与求解器运行文件；在 Ubuntu 本地读取，不复制资源。"""
    sim = Path(team) / "sckd_sim"
    paths = {"binary": Path(binary) if binary is not None else sim / "build-predictor/m2_intercept", "models": sim / "models",
             "cpp_sources": sim / "cpp", "cmake": sim / "CMakeLists.txt"}
    if mode == "nmpc6":
        paths.update({f"nmpc6/{p.name}": p for p in sorted((sim / "third_party/nmpc6").glob("*.so"))})
        if "nmpc6/nlp.so" not in paths:
            raise FileNotFoundError(sim / "third_party/nmpc6/nlp.so")
    return file_resources(paths)


def resolved_runtime_config(log):
    """新二进制提供的实际解析值；历史二进制缺此行时明确返回 None。"""
    prefix = "[resolved config] "
    values = [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    if not values:
        return None
    if any(not isinstance(value, dict) or value != values[0] for value in values):
        raise ValueError("conflicting or invalid resolved runtime configurations")
    return values[0]


@dataclass(frozen=True)
class ExperimentSpec:
    runner: str
    command: tuple[str, ...]
    cwd: str
    parameters: dict
    environment: dict
    resources: dict
    service_command: tuple[str, ...] = ()

    def resolved(self):
        # JSON round trip rejects non-serializable/nonfinite values and makes a detached snapshot.
        return json.loads(_json({"schema": SCHEMA, "runner": self.runner,
            "command": list(self.command), "cwd": self.cwd, "parameters": self.parameters,
            "environment": self.environment, "resources": self.resources,
            "service_command": list(self.service_command),
            "runtime": {"system": platform.system(), "machine": platform.machine(),
                        "runner_python": sys.version}}))


def _identity(spec):
    return hashlib.sha256(_json(spec).encode()).hexdigest()


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class RunAttempt:
    """保持旧结果字段；新增 experiment 包含完整请求、内容身份及输出哈希。

    旧无身份清单的成功结果不能证明配置相同，故拒绝自动复用，也不覆盖。
    --retry-incomplete 只归档失败/中断尝试；不会覆盖任何已有完成文件。
    """
    def __init__(self, destination, spec, *, completion="run.json", retry_incomplete=False):
        self.destination = Path(destination)
        self.spec = spec.resolved()
        self.identity = _identity(self.spec)
        self.completion = completion
        self.retry = retry_incomplete
        self.existing = None
        self.started = False
        self.finished = False
        self.lock = None

    def __enter__(self):
        self.destination.mkdir(parents=True, exist_ok=True)
        self.lock = (self.destination / ".run.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._prepare()
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def _prepare(self):
        self._verify_resources()
        done = self.destination / self.completion
        if done.exists():
            result = json.loads(done.read_text())
            metadata = result.get("experiment")
            if not metadata or metadata.get("schema") != SCHEMA or metadata.get("status") != "completed":
                raise FileExistsError(f"legacy result has no verifiable resolved spec; preserved: {done}; use a new output directory")
            if metadata.get("identity") != self.identity or metadata.get("spec") != self.spec:
                raise FileExistsError(f"completed result differs in input, binary, resources or configuration; preserved: {done}; use a new output directory")
            metrics = {key: value for key, value in result.items() if key != "experiment"}
            if metadata.get("result_sha256") != _identity(metrics):
                raise ValueError(f"completed result fields changed; preserved: {done}")
            artifacts = metadata.get("artifacts")
            if not artifacts:
                raise ValueError(f"result has no output hashes; preserved: {done}")
            for relative, expected in artifacts.items():
                path = self.destination / relative
                if not path.is_file() or digest(path) != expected:
                    raise ValueError(f"completed output missing/changed; preserved: {path}; use a new output directory")
            self.existing = result
            return self
        leftovers = [p for p in self.destination.iterdir() if p.name not in ("attempts", ".run.lock")]
        if leftovers:
            if not self.retry:
                raise FileExistsError(f"incomplete run preserved: {self.destination}; inspect it, then use --retry-incomplete or a new output directory")
            archive = self.destination / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
            archive.mkdir(parents=True)
            for path in leftovers:
                path.rename(archive / path.name)
        self.started_at = _stamp()
        write_json(self.destination / "run_manifest.json", self._manifest("running"), exclusive=True)
        self.started = True
        return self

    def _manifest(self, status, **extra):
        return {"schema": SCHEMA, "identity": self.identity, "spec": self.spec,
                "status": status, "started_at_utc": self.started_at, **extra}

    def _verify_resources(self):
        for resource in self.spec["resources"].values():
            if digest(resource["path"]) != resource["sha256"]:
                raise RuntimeError(f"resource changed during run: {resource['path']}")

    def complete(self, result):
        if not self.started or self.finished:
            raise RuntimeError("attempt is not active")
        self._verify_resources()
        artifacts = {str(path.relative_to(self.destination)): digest(path)
                     for path in sorted(self.destination.rglob("*"))
                     if path.is_file() and "attempts" not in path.relative_to(self.destination).parts
                     and path.name not in ("run_manifest.json", ".run.lock", self.completion)}
        if not artifacts:
            raise RuntimeError("refusing to complete an experiment without recorded outputs")
        result = {**result, "experiment": self._manifest("completed", completed_at_utc=_stamp(),
                  artifacts=artifacts, result_sha256=_identity(result))}
        write_json(self.destination / self.completion, result, exclusive=True)
        self.finished = True
        write_json(self.destination / "run_manifest.json", result["experiment"])
        return result

    def __exit__(self, kind, error, traceback):
        try:
            if self.started and not self.finished:
                failure = error if error is not None else RuntimeError("attempt exited without completion")
                write_json(self.destination / "run_manifest.json", self._manifest("failed", ended_at_utc=_stamp(),
                           error={"type": type(failure).__name__, "message": str(failure)}))
                if error is None:
                    raise failure
        finally:
            if self.lock is not None:
                self.lock.close()
                self.lock = None
        return False


def socket_path(team, prefix):
    path = Path(team) / ".tmp" / f"{prefix}-{uuid.uuid4().hex[:12]}.sock"
    if len(os.fsencode(path)) > 100:
        raise ValueError(f"Unix socket path is too long: {path}")
    return path


def _stop(process, grace):
    # Every child owns a new session; descendants are stopped with its process group.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def positive_timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def run_logged(command, *, cwd, env, log, timeout):
    timeout = positive_timeout(timeout)
    with Path(log).open("x") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout)
        except BaseException:
            _stop(process, 3)
            raise
    return subprocess.CompletedProcess(command, returncode)


@contextmanager
def socket_service(command, *, socket, cwd, env, log, startup_timeout=10, stop_timeout=3):
    """等待 socket 文件就绪；不探测连接，避免占用只接受一个客户端的缓存服务。"""
    startup_timeout = positive_timeout(startup_timeout)
    stop_timeout = positive_timeout(stop_timeout)
    socket = Path(socket)
    socket.parent.mkdir(parents=True, exist_ok=True)
    if socket.exists() or socket.is_symlink():
        raise FileExistsError(f"refusing to replace existing socket: {socket}")
    process = None
    owner = None
    try:
        with Path(log).open("x") as stream:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"socket service exited with {process.returncode}; see {log}")
                    if socket.exists():
                        info = socket.lstat()
                        if not stat.S_ISSOCK(info.st_mode):
                            raise RuntimeError(f"service path is not a socket: {socket}")
                        owner = (info.st_dev, info.st_ino)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"socket service startup timed out after {startup_timeout}s; see {log}")
                    time.sleep(min(.02, max(0, deadline - time.monotonic())))
                yield process
            finally:
                _stop(process, stop_timeout)
    finally:
        if socket.exists():
            info = socket.lstat()
            if stat.S_ISSOCK(info.st_mode) and (owner is None or owner == (info.st_dev, info.st_ino)):
                socket.unlink()
