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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        help="original team archive; needed only if the bundled OSQP library is missing",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("sckd_sim/build-predictor"),
        help="build directory relative to the project root (default: %(default)s)",
    )
    args = parser.parse_args()

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("requires Linux x86_64; bundled third-party libraries are platform-specific")

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
        "https://pypi.org/simple",
        "mujoco==3.14.0",
        "casadi==3.8.0",
        "numpy",
    )
    for name in ("mujoco", "casadi"):
        link_package(name, python)

    source = ROOT / "sckd_sim"
    lib = source / "third_party/osqp/libosqp.a"
    restore_osqp(lib, args.archive)
    if hashlib.sha256(lib.read_bytes()).hexdigest() != OSQP_SHA256:
        raise RuntimeError("OSQP hash mismatch")

    nlp = source / "third_party/nmpc6/nlp.so"
    if not nlp.exists():
        run("gcc", "-fPIC", "-shared", "-O3", source / "third_party/nmpc6/nlp.c", "-o", nlp, "-lm")

    compiler = shutil.which("g++-13") or shutil.which("g++")
    if compiler is None:
        raise SystemExit("no C++ compiler found; install g++-13 or another C++17 compiler")
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
    run("cmake", "--build", build, "-j8")
    run("ctest", "--test-dir", build, "--output-on-failure")


if __name__ == "__main__":
    main()
