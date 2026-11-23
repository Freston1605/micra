"""Build an offline wheelhouse from the locked environment.

The lockfile is the single source of truth for every package except the deep
learning framework. Torch and torchvision are replaced by the CUDA 12.6
wheels from the pinned PyTorch index: those are the builds that still compile
the V100 ``sm_70`` kernel set. The default public index is never used for
those two packages.

The CUDA runtime below them is replaced with them. The lockfile pins the
default-index tree (torch 2.13 asks for CUDA 13), while the cu126 wheels
declare CUDA 12.6 exactly; carrying both pins is a resolution conflict. The
builder therefore drops every ``cuda-*``/``nvidia-*`` pin from the default
list and resolves that tree from the cu126 wheel's own metadata, with the rest
of the lockfile passed as a constraint file so numpy and friends cannot drift.

Everything the compute node needs is downloaded here, on a machine with
internet access, and verified by a ``SHA256SUMS`` manifest. Installation on
the node is then purely offline::

    uv pip install --no-index --find-links dist/wheelhouse \
        -r dist/wheelhouse/requirements.txt

Usage::

    python tools/hpc/build_wheelhouse.py [--wheelhouse dist/wheelhouse] [--dry-run]

``--dry-run`` prints the plan without downloading anything; it is safe to run
offline and is what the test suite exercises.

The builder never installs into the environment that invoked it: it downloads,
builds and verifies, and the image (or the node) does the offline install. A
self-check that ran `uv pip install` here would need a virtual environment
inside the image's build stage and, when it found none, could write into an
unrelated one.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UV_VERSION = "0.13.7"
PIP_VERSION = "26.3"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
PYPI_INDEX = "https://pypi.org/simple"
CUDA_PACKAGES = ("torch==2.13.0+cu126", "torchvision==0.28.0+cu126")
REPLACED_PACKAGES = frozenset({"torch", "torchvision"})
#: The CUDA runtime tree the framework wheel selects. It is listed by name, not
#: by prefix: a future `nvidia-*` or `cuda-*` package that is *not* part of this
#: tree must keep its lockfile pin, and a prefix filter would silently drop it.
#: tests/test_hpc_environment.py proves this set equals the closure of
#: torch/torchvision in uv.lock, so a framework bump that changes the tree
#: fails there instead of producing a wheelhouse with a missing pin.
CUDA_RUNTIME_PACKAGES = frozenset(
    {
        "cuda-bindings",
        "cuda-pathfinder",
        "cuda-toolkit",
        "nvidia-cublas",
        "nvidia-cuda-cupti",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cufft",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-cusparselt-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvjitlink",
        "nvidia-nvshmem-cu13",
        "nvidia-nvtx",
    }
)
#: A framework pin is `name==version+tag`, and the tag must be the index's own.
_CUDA_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\d[^+]*)\+(?P<tag>[A-Za-z0-9.]+)$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+")
DEFAULT_WHEELHOUSE = Path("dist") / "wheelhouse"


@dataclass(frozen=True)
class Plan:
    """The concrete steps needed to build and verify a wheelhouse."""

    wheelhouse: Path
    default_requirements: tuple[str, ...]
    cuda_requirements: tuple[str, ...]
    cuda_index_args: tuple[str, ...]
    steps: tuple[tuple[str, ...], ...]

    @property
    def requirements_file(self) -> Path:
        return self.wheelhouse / "requirements.txt"

    @property
    def checksum_file(self) -> Path:
        return self.wheelhouse / "SHA256SUMS"


def cuda_tag(index: str = CUDA_INDEX) -> str:
    """The local version tag an index serves, e.g. ``cu126`` for ``/whl/cu126``."""
    return index.rstrip("/").rsplit("/", 1)[-1]


def check_framework_pins(pins: tuple[str, ...], index: str = CUDA_INDEX) -> None:
    """Refuse a framework pin that is not an exact build of the given index.

    A floating requirement, or a ``+cu126`` pin aimed at some other index, is
    only discovered after a multi-gigabyte download has failed, or worse, after
    the wrong build reached the wheelhouse.
    """
    expected = cuda_tag(index)
    for pin in pins:
        match = _CUDA_PIN.match(pin)
        if match is None:
            raise ValueError(f"framework pin {pin!r} is not an exact `name==version+tag`")
        if match["tag"] != expected:
            raise ValueError(
                f"framework pin {pin!r} does not belong to {index} (expected +{expected})"
            )


def split_requirements(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep the locked pins, minus the framework and its CUDA runtime tree."""
    default: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = _NAME.match(line)
        name = match.group(0).lower() if match else ""
        if name in REPLACED_PACKAGES or name in CUDA_RUNTIME_PACKAGES:
            continue
        default.append(line)
    return tuple(default), CUDA_PACKAGES


def make_plan(requirements_text: str, wheelhouse: Path) -> Plan:
    """Turn an exported requirements list into ordered build steps."""
    default, cuda = split_requirements(requirements_text)
    check_framework_pins(cuda)
    wheelhouse = Path(wheelhouse)
    exported = wheelhouse / "lock-requirements.txt"
    pip = (
        "uv",
        "run",
        "--no-sync",
        "--with",
        f"pip=={PIP_VERSION}",
        "python",
        "-m",
        "pip",
        "download",
    )
    # The lockfile already contains every transitive pin outside the CUDA tree,
    # so the default download is dependency-free: resolving again would pull
    # newer numpy/Pillow wheels that contradict the lock. The CUDA download does
    # resolve, because the tree it needs is the one its own wheel declares; the
    # exported lockfile is passed as the constraint file, so every other pin
    # stays where the lockfile put it.
    #
    # Both indexes are named because the framework needs one and its
    # dependencies need the other, but the pin picks the source: `+cu126` is a
    # local version tag no default-index wheel carries, so PyPI cannot satisfy
    # `torch==2.13.0+cu126`. `verify_cuda_runtime` then proves the bytes match
    # that pin before the manifest is written.
    index_args = ("--index-url", CUDA_INDEX)
    steps = (
        (
            "uv",
            "export",
            "--frozen",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements-txt",
            "-o",
            str(exported),
        ),
        ("write", str(exported), "locked pins without default torch/torchvision"),
        (*pip, "--no-deps", "--dest", str(wheelhouse), "-r", str(exported)),
        (
            *pip,
            *index_args,
            "--extra-index-url",
            PYPI_INDEX,
            "--dest",
            str(wheelhouse),
            "-c",
            str(exported),
            *cuda,
        ),
        # The resolver is the only thing standing between a pin and the tree it
        # drags in, so the download is checked by name before anything is
        # published: a CUDA 13 runtime, a plain framework wheel or a missing one
        # fails here instead of on the node.
        ("verify", str(wheelhouse), "framework is +cu126 and the runtime is CUDA 12.6"),
        # The project's own wheel is built here, where the build backend and the
        # network are available, and ships in the wheelhouse. A compute node
        # then installs it with --no-index and never needs hatchling, Babel or
        # the source tree; building it there also required NOTICE, which the
        # force-include in pyproject.toml makes mandatory.
        ("uv", "build", "--wheel", "--out-dir", str(wheelhouse)),
        ("write", str(wheelhouse / "requirements.txt"), "locked pins + cu126 wheels"),
        ("write", str(wheelhouse / "SHA256SUMS"), "digest of every wheel in the directory"),
    )
    return Plan(
        wheelhouse=wheelhouse,
        default_requirements=default,
        cuda_requirements=cuda,
        cuda_index_args=index_args,
        steps=steps,
    )


def _wheel_fields(filename: str) -> tuple[str, str]:
    """The normalized name and the version of a wheel file name."""
    name, _, rest = filename.partition("-")
    version, _, _ = rest.partition("-")
    return name.replace("_", "-").lower(), version


#: A CUDA 13 flavour, as it appears in a wheel name (``..._cu13``).
_CUDA13 = re.compile(r"(?:^|[-_])cu13(?:[-_]|$)")


def verify_cuda_runtime(wheelhouse: Path) -> tuple[str, ...]:
    """Problems in a downloaded wheelhouse, empty when it is the cu126 stack.

    The resolver is the only thing between a pin and the tree it drags in, and
    its two failure modes are silent until a job dies on the node: the
    framework came from the default index, or a CUDA 13 runtime wheel won the
    resolution. Both are visible in file names, so this check needs no import
    of the wheels and no network.
    """
    expected = {pin.partition("==")[0]: pin.partition("==")[2] for pin in CUDA_PACKAGES}
    problems: list[str] = []
    downloaded = set()
    cuda_runtime_wheels = 0
    for wheel in sorted(wheelhouse.glob("*.whl")):
        name, version = _wheel_fields(wheel.name)
        if name in expected:
            downloaded.add(name)
            if version != expected[name]:
                problems.append(f"{wheel.name}: expected {name}=={expected[name]}")
            continue
        if not name.startswith(("cuda-", "nvidia-")):
            continue
        if _CUDA13.search(name) or version.partition(".")[0] == "13":
            problems.append(f"{wheel.name}: CUDA 13 build, not the cu126 runtime")
        else:
            cuda_runtime_wheels += 1
    problems.extend(f"{name}: not downloaded" for name in sorted(set(expected) - downloaded))
    if not cuda_runtime_wheels:
        problems.append("no CUDA 12 runtime wheel was downloaded")
    return tuple(problems)


def _export_lockfile() -> str:
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-emit-project",
        "--no-hashes",
        "--format",
        "requirements-txt",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "uv export failed")
    return result.stdout


def _write_index(plan: Plan) -> None:
    plan.wheelhouse.mkdir(parents=True, exist_ok=True)
    header = [
        "# Generated from uv.lock by tools/hpc/build_wheelhouse.py; do not edit.",
        "# Every pin comes from the lockfile except the two CUDA 12.6 wheels, which",
        "# replace its default-index torch/torchvision pins. The CUDA runtime below",
        "# them is resolved from those wheels' own metadata and is present in this",
        "# directory; it is deliberately not pinned here, because the lockfile pins",
        "# the tree of the default-index build. The digest of every wheel, including",
        "# that one, is in SHA256SUMS.",
        "# The project's own wheel sits beside this file and is installed separately:",
        "#   uv pip install --no-index --no-deps micra-*.whl",
        "# Install offline with: uv pip install --no-index --find-links <dir> -r <dir>/requirements.txt",
    ]
    pins = [*plan.default_requirements, *plan.cuda_requirements]
    plan.requirements_file.write_text("\n".join([*header, *pins]) + "\n", encoding="utf-8")


def _write_checksums(plan: Plan) -> None:
    # The wheelhouse stage has no installed dependencies yet: importing the
    # ``polyx`` package would pull in cv2 through its scientific stack. Load
    # the standalone hashing module by path instead, with no heavy imports.
    import importlib.util

    path = ROOT / "src" / "polyx" / "hpc" / "hashing.py"
    spec = importlib.util.spec_from_file_location("polyx_hpc_hashing", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    hashing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hashing)

    wheels = sorted(plan.wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheels were downloaded into {plan.wheelhouse}")
    hashing.write_manifest(plan.checksum_file, wheels)
    problems = hashing.verify_manifest(plan.wheelhouse, plan.checksum_file)
    if problems:
        raise RuntimeError("wheelhouse verification failed: " + "; ".join(problems))


def _run(step: tuple[str, ...]) -> int:
    print("+ " + " ".join(step), flush=True)
    if step[0] == "write":
        return 0
    return subprocess.run(step, cwd=ROOT, check=False).returncode


def _write_pins(path: Path, pins: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pins) + "\n", encoding="utf-8")


def _apply_step(plan: Plan, step: tuple[str, ...]) -> int:
    """Run one plan step; a non-zero result stops the build."""
    kind = step[0]
    if kind == "write":
        target = Path(step[1])
        if target == plan.wheelhouse / "lock-requirements.txt":
            _write_pins(target, plan.default_requirements)
        elif target == plan.requirements_file:
            _write_index(plan)
        elif target == plan.checksum_file:
            _write_checksums(plan)
        print("+ " + " ".join(step), flush=True)
        return 0
    if kind == "verify":
        problems = verify_cuda_runtime(plan.wheelhouse)
        print("+ " + " ".join(step), flush=True)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1 if problems else 0
    return _run(step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline wheelhouse from uv.lock")
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; download nothing")
    args = parser.parse_args(argv)

    try:
        requirements_text = _export_lockfile()
    except (OSError, RuntimeError) as exc:
        print(f"cannot read uv.lock: {exc}", file=sys.stderr)
        return 1
    plan = make_plan(requirements_text, args.wheelhouse)

    if args.dry_run:
        for step in plan.steps:
            print("+ " + " ".join(step))
        print(f"default pins: {len(plan.default_requirements)}")
        print(f"cuda wheels: {', '.join(plan.cuda_requirements)}")
        print(
            f"offline install: uv pip install --no-index --find-links {plan.wheelhouse} -r {plan.requirements_file}"
        )
        print(f"project wheel: uv pip install --no-index --no-deps {plan.wheelhouse}/micra-*.whl")
        return 0

    for step in plan.steps:
        code = _apply_step(plan, step)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
