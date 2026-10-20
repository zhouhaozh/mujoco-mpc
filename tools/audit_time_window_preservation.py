"""只读核对时间窗口实验部署、原资源和 Git；新报告必须写在冻结证据根之外。

baseline_manifest 仅记录原源码和两个旧二进制；模型/NLP 用最早默认等价
input_manifest 补证，35条原输入和原 h1500 缓存用最早 stage1 搜索清单补证。
不 reset、add、commit、push，不导入仿真、绘图或训练依赖。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCHEMA = "TIME_WINDOW_PRESERVATION_AUDIT/1"


def sha256_file(path):
    path = Path(path)
    before = path.stat()
    if not path.is_file():
        raise ValueError("not a regular file")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after):
        raise RuntimeError("file changed while hashing")
    return value.hexdigest()


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("manifest contains an invalid SHA-256")
    return value


def absolute_file(value, team):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("manifest resource paths must be absolute")
    path = Path(os.path.abspath(value))
    if not path.is_relative_to(team):
        raise ValueError(f"manifest resource is outside the requested team root: {path}")
    return str(path)


def read_document(path, evidence):
    path = Path(path)
    payload = path.read_bytes()
    evidence.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    return json.loads(payload)


def baseline_copy(root, team, source):
    relative = Path(source).relative_to(team)
    aliases = {Path("sckd_sim/build-predictor/m2_intercept"): "default_binary",
               Path("sckd_sim/build-modularization/m2_intercept"): "modular_binary"}
    return root / "baseline" / aliases.get(relative, relative)


def inspect_file(path, expected, *, baseline_hash=None, deployed=False, **metadata):
    result = {"path": str(path), "expected_sha256": expected,
              "baseline_sha256": baseline_hash, **metadata}
    try:
        actual = sha256_file(path)
        status = ("authorized_new_file" if baseline_hash is None else "authorized_deployment") if deployed else "unchanged"
        result.update(sha256=actual, status=status if actual == expected else "hash_mismatch")
    except FileNotFoundError:
        result.update(sha256=None, status="missing")
    except (OSError, ValueError, RuntimeError) as error:
        result.update(sha256=None, status="unreadable", error=str(error))
    return result


def git_state(team, expected_branch, expected_head=None):
    state = {"expected_branch": expected_branch, "expected_head": expected_head, "errors": []}
    def command(arguments):
        try:
            result = subprocess.run(["git", "--no-optional-locks", "-C", str(team), *arguments],
                capture_output=True, text=True, timeout=20,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        except (OSError, subprocess.TimeoutExpired) as error:
            state["errors"].append(str(error))
            return None
        if result.returncode:
            state["errors"].append({"arguments": arguments, "returncode": result.returncode,
                                     "stderr": result.stderr.strip()})
            return None
        return result.stdout
    branch = command(["symbolic-ref", "--quiet", "--short", "HEAD"])
    head = command(["rev-parse", "--verify", "HEAD"])
    staged = command(["diff", "--cached", "--name-only", "-z", "--no-ext-diff", "--ignore-submodules=none", "HEAD", "--"])
    state.update(branch=None if branch is None else branch.strip(),
                 head=None if head is None else head.strip(),
                 staged_paths=None if staged is None else [name for name in staged.split("\0") if name])
    state["branch_matches"] = state["branch"] == expected_branch
    state["head_matches"] = expected_head is None or state["head"] == expected_head
    state["staging_empty"] = state["staged_paths"] == []
    state["passed"] = not state["errors"] and state["branch_matches"] and state["head_matches"] and state["staging_empty"]
    return state


def audit(root, team, *, expected_branch, expected_head=None, git_reader=git_state):
    root, team = Path(root).resolve(), Path(team).resolve()
    evidence, issues, expected = [], [], {}
    baseline = read_document(root / "baseline_manifest.json", evidence)
    deployment = read_document(root / "deployment_manifest.json", evidence)
    original = baseline.get("files") if isinstance(baseline, dict) else None
    if not isinstance(original, dict) or not original:
        raise ValueError("baseline_manifest.json requires a nonempty files mapping")
    original = {absolute_file(path, team): checked_hash(value) for path, value in original.items()}
    if not isinstance(deployment, list):
        raise ValueError("deployment_manifest.json requires a list")
    deployed = {}
    for entry in deployment:
        path = absolute_file(entry["path"], team)
        if path in deployed:
            raise ValueError(f"duplicate deployment path: {path}")
        deployed[path] = {"sha256": checked_hash(entry["sha256"]), "before": entry.get("before")}
        before = entry.get("before")
        if before is not None:
            checked_hash(before)
        if before != original.get(path):
            issues.append({"kind": "deployment_before_mismatch", "path": path,
                           "baseline_sha256": original.get(path), "deployment_before": before})

    def add(path, sha, category, origin):
        path, sha = absolute_file(path, team), checked_hash(sha)
        if path in expected:
            previous = expected[path]
            if previous["sha256"] != sha:
                issues.append({"kind": "conflicting_evidence", "path": path,
                               "earlier_sha256": previous["sha256"], "later_sha256": sha,
                               "earlier_origins": list(previous["origins"]), "later_origin": origin})
            previous["origins"].append(origin)
            if category not in previous["categories"]:
                previous["categories"].append(category)
        else:
            expected[path] = {"sha256": sha, "categories": [category], "origins": [origin]}

    for path, sha in original.items():
        category = "default_binary" if path.endswith("/build-predictor/m2_intercept") else (
            "old_modular_binary" if path.endswith("/build-modularization/m2_intercept") else "original_source")
        if category in ("default_binary", "old_modular_binary") and path in deployed:
            issues.append({"kind": "protected_binary_listed_for_deployment", "path": path})
        expected_sha = deployed[path]["sha256"] if path in deployed and category == "original_source" else sha
        add(path, expected_sha, category, "baseline_manifest.json + explicit deployment")
    for path, entry in deployed.items():
        if path not in original:
            add(path, entry["sha256"], "deployed_new_source", "deployment_manifest.json")

    model_doc = read_document(root / "default_equivalence/input_manifest.json", evidence)
    if not isinstance(model_doc, dict) or not isinstance(model_doc.get("resources"), dict):
        raise ValueError("default equivalence input manifest requires a resources mapping")
    model_resources = model_doc.get("resources", {})
    models, solvers = set(), set()
    for label, resource in model_resources.items():
        if label.startswith(("model/", "nmpc6/")) or label == "scene":
            category = "model_resource" if not label.startswith("nmpc6/") else "original_nlp"
            add(resource["path"], resource["sha256"], category, "default_equivalence/input_manifest.json:" + label)
            (solvers if category == "original_nlp" else models).add(resource["path"])
    frozen = read_document(root / "stage1/search_manifest.json", evidence)
    if frozen.get("schema") != "TIME_WINDOW_SEARCH_MANIFEST/1":
        raise ValueError("the original stage1 search manifest is unsupported")
    inputs, caches = set(), set()
    input_metadata, cache_metadata = 0, 0
    for label, resource in frozen.get("resources", {}).items():
        category = None
        if label.startswith("input/"):
            inputs.add(label.split("/", 1)[1]);category = "original_input"
        elif label.startswith("frozen_cache/h1500/"):
            caches.add(label.split("/", 2)[2]);category = "original_h1500_cache"
        elif label == "input_manifest":
            input_metadata += 1;category = "original_input_manifest"
        elif label == "cache_manifest/h1500":
            cache_metadata += 1;category = "original_h1500_cache_manifest"
        if category:
            add(resource["path"], resource["sha256"], category, "stage1/search_manifest.json:" + label)
    for mode, resources in frozen.get("runtime_resources", {}).items():
        for label, resource in resources.items():
            if label.startswith(("models/", "nmpc6/")):
                category = "model_resource" if label.startswith("models/") else "original_nlp"
                add(resource["path"], resource["sha256"], category, "stage1/search_manifest.json:" + mode + "/" + label)
    split = frozen.get("split", {})
    sample_ids = split.get("discovery", []) + split.get("confirmation", [])
    coverage = {"baseline_files": len(original), "deployment_files": len(deployed),
        "model_resources": len(models), "original_nlp_files": len(solvers),
        "original_input_samples": len(inputs), "original_h1500_cache_samples": len(caches),
        "original_input_manifests": input_metadata, "original_cache_manifests": cache_metadata,
        "sample_ids": sorted(inputs)}
    coverage["passed"] = bool(models and solvers and input_metadata == 1 and cache_metadata == 1
        and len(sample_ids) == len(set(sample_ids)) == 35 and inputs == caches == set(sample_ids)
        and any(path.endswith("/build-predictor/m2_intercept") for path in original)
        and any(path.endswith("/build-modularization/m2_intercept") for path in original))
    if not coverage["passed"]:
        issues.append({"kind": "incomplete_protection_evidence", "coverage": coverage})

    checks = []
    for path, resource in sorted(expected.items()):
        checks.append(inspect_file(path, resource["sha256"], baseline_hash=original.get(path),
            deployed=path in deployed, categories=resource["categories"], origins=resource["origins"]))
    for path, sha in sorted(original.items()):
        checks.append(inspect_file(baseline_copy(root, team, path), sha,
            categories=["immutable_baseline_copy"], original_path=path, origins=["baseline_manifest.json"]))
    failures = [entry for entry in checks if entry["status"] not in
                ("unchanged", "authorized_deployment", "authorized_new_file")]
    git = git_reader(team, expected_branch, expected_head)
    counts = dict(Counter(entry["status"] for entry in checks))
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "host_platform": sys.platform, "evidence_root": str(root), "team": str(team),
        "passed": not issues and not failures and git["passed"], "coverage": coverage,
        "counts": counts, "manifest_evidence": evidence, "git": git,
        "issues": issues, "failures": failures,
        "authorized_deployments": [entry for entry in checks if entry["status"].startswith("authorized_")],
        "checks": checks,
        "boundaries": ["baseline_manifest only predates the explicit source/binary deployment; it has no model/input/cache hashes",
            "model/NLP evidence starts at default_equivalence; original35 input/cache evidence starts at stage1 manifest creation",
            "expected Git branch/head are explicit audit arguments; the baseline manifest did not record Git state",
            "only listed resources and their baseline copies are checked; generated/new unlisted files are outside this audit",
            "read-only snapshot audit; no Git or protected resource changes are performed"]}


def write_report(path, report):
    path = Path(path).resolve()
    root = Path(report["evidence_root"]).resolve()
    protected = {Path(entry["path"]).resolve() for entry in report["checks"] + report["manifest_evidence"]}
    if path == root or path.is_relative_to(root) or path in protected:
        raise ValueError("audit output must be outside the frozen evidence root and protected resources")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--team", required=True, type=Path)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-head")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.root, args.team, expected_branch=args.expected_branch, expected_head=args.expected_head)
        write_report(args.out, report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"preservation audit could not complete: {error}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"passed": report["passed"], "counts": report["counts"], "issues": report["issues"],
        "failures": report["failures"], "git": report["git"], "out": str(args.out.resolve())}, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
