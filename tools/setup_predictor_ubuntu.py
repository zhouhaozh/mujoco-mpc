"""Install project-local Linux x86_64 dependencies, build, and run CTest."""

import argparse
import hashlib
import platform
import shutil
import subprocess
import tarfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OSQP_ARCHIVE_MEMBER = "sim/sckd_sim/third_party/osqp/libosqp.a"
OSQP_SHA256 = "a00af25b409510a548e5125544fef402b04d0eb494f6ff853a2fbe5c71700c39"


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def project_path(path):
    return path if path.is_absolute() else ROOT / path


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def link_package(name, python):
    package = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                f"import {name}; from pathlib import Path; "
                f"print(Path({name}.__file__).resolve().parent)",
            ],
            text=True,
        ).strip()
    )
    target = ROOT / "deps" / name
    target.mkdir(parents=True, exist_ok=True)
    for label, source in (("include", package / "include"), ("lib", package)):
        link = target / label
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"refusing to replace {link}")
        link.symlink_to(source, target_is_directory=True)
    if name == "mujoco" and not (package / "libmujoco.so").exists():
        (package / "libmujoco.so").symlink_to("libmujoco.so.3.14.0")


def restore_osqp(lib, archive):
    if lib.exists():
        return
    if archive is None:
        raise SystemExit(
            f"missing {lib}; restore it from this repository or pass "
            "--archive /path/to/original-team-archive.tar.xz"
        )
    with tarfile.open(project_path(archive)) as bundle:
        member = bundle.extractfile(OSQP_ARCHIVE_MEMBER)
        if member is None:
            raise RuntimeError(f"missing archive member {OSQP_ARCHIVE_MEMBER}")
        data = member.read()
    if hashlib.sha256(data).hexdigest() != OSQP_SHA256:
        raise RuntimeError("unexpected OSQP binary")
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(
        description="创建项目 Python 环境、编译仿真程序并运行 CTest。"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="原团队归档；仅在仓库内缺少 OSQP 静态库时需要",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("sckd_sim/build-predictor"),
        help="相对项目根目录的构建目录（默认：%(default)s）",
    )
    parser.add_argument(
        "--jobs",
        type=positive_integer,
        default=2,
        help="并行编译任务数（默认：%(default)s）",
    )
    parser.add_argument(
        "--index-url",
        default="https://pypi.org/simple",
        help="Python 包索引地址（默认：%(default)s）",
    )
    args = parser.parse_args()

    if not args.index_url:
        parser.error("--index-url 不能为空")

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("仅支持 Linux x86_64；仓库内第三方二进制库与平台相关")

    source = ROOT / "sckd_sim"
    lib = source / "third_party/osqp/libosqp.a"
    restore_osqp(lib, args.archive)
    if hashlib.sha256(lib.read_bytes()).hexdigest() != OSQP_SHA256:
        raise RuntimeError("OSQP hash mismatch")

    compiler = shutil.which("g++-13") or shutil.which("g++")
    if compiler is None:
        raise SystemExit("找不到 C++ 编译器；请安装 g++-13 或其他 C++17 编译器")
    for command in ("gcc", "cmake", "ctest"):
        if shutil.which(command) is None:
            raise SystemExit(f"找不到构建工具 {command}；请先安装系统依赖")

    requirements = ROOT / "requirements-sim.txt"
    if not requirements.is_file():
        raise SystemExit(f"缺少 Python 依赖锁定文件：{requirements}")

    environment = ROOT / ".venv"
    if not (environment / "bin/python").exists():
        venv.create(environment, with_pip=True)
    python = environment / "bin/python"
    run(
        python,
        "-m",
        "pip",
        "install",
        "--index-url",
        args.index_url,
        "--requirement",
        requirements,
    )
    run(python, "-m", "pip", "check")
    for name in ("mujoco", "casadi"):
        link_package(name, python)

    nlp = source / "third_party/nmpc6/nlp.so"
    if not nlp.exists():
        run("gcc", "-fPIC", "-shared", "-O3", source / "third_party/nmpc6/nlp.c", "-o", nlp, "-lm")

    build = project_path(args.build_dir)
    run(
        "cmake",
        "-S",
        source,
        "-B",
        build,
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DMUJOCO_DIR={ROOT / 'deps/mujoco'}",
        f"-DCASADI_DIR={ROOT / 'deps/casadi'}",
        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1",
    )
    run("cmake", "--build", build, f"-j{args.jobs}")
    run("ctest", "--test-dir", build, "--output-on-failure")


if __name__ == "__main__":
    main()
