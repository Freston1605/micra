"""The environment contract for the V100 training stack.

These tests drive the public reporting surface, not the private probe helpers.
A fake torch module stands in for the GPU: it lets the "hardware present" path
be exercised offline, while the default fake has no CUDA so the honest
"absent" path is the one a GPU-less machine actually takes. The real hardware
smoke check is ticket #199 on OCEANO, not this suite.
"""

from __future__ import annotations

import contextlib
import subprocess
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from polyx.hpc import check_env, environment, hashing, weights

ROOT = Path(__file__).resolve().parents[1]

FLOAT16 = object()


class _Tensor:
    """Just enough of a torch tensor for the AMP probe."""

    def __init__(self, dtype: object, finite: bool = True) -> None:
        self.dtype = dtype
        self.finite = finite

    def __matmul__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.dtype, self.finite and other.finite)


class _BoolTensor:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _BoolTensor:
        return self

    def item(self) -> bool:
        return self.value


def _fake_torch(
    *,
    available: bool = True,
    architectures: tuple[str, ...] = ("sm_50", "sm_70", "sm_80"),
    cuda_version: str | None = "12.6",
    finite: bool = True,
    compiled_flags: str | None = "sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90",
) -> SimpleNamespace:
    """A torch-shaped object whose answers are controlled by the test."""
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_arch_list=lambda: list(architectures) if available else [],
        get_device_name=lambda _index: "Tesla V100-PCIE-32GB" if available else "",
        synchronize=lambda: None,
    )

    def randn(_shape: tuple[int, ...], *, device: str, dtype: object) -> _Tensor:
        assert device == "cuda"
        return _Tensor(dtype, finite)

    def autocast(device_type: str, *, dtype: object) -> object:
        assert device_type == "cuda"
        assert dtype is FLOAT16
        return contextlib.nullcontext()

    internal = SimpleNamespace(_cuda_getArchFlags=lambda: compiled_flags)
    return SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda=cuda_version),
        float16=FLOAT16,
        randn=randn,
        autocast=autocast,
        isfinite=lambda tensor: _BoolTensor(tensor.finite),
        _C=internal,
    )


# ── CUDA / architecture / AMP reporting ──


def test_no_gpu_reports_absence_and_does_not_fake_an_amp_pass() -> None:
    report = environment.probe(_fake_torch(available=False))

    assert report.cuda_available is False
    assert report.amp_status == environment.AMP_NOT_RUN
    assert report.amp_dtype is None
    assert report.ok is False
    assert any("CUDA is not available" in problem for problem in report.problems)

    lines = "\n".join(report.lines())
    assert "CUDA available: no" in lines
    assert "Reported architectures: (none)" in lines
    assert "AMP pass: not run" in lines
    assert "Result: FAIL" in lines


def test_gpu_with_v100_architecture_and_amp_passes() -> None:
    report = environment.probe(_fake_torch())

    assert report.cuda_available is True
    assert "sm_70" in report.architectures
    assert report.cuda_version == "12.6"
    assert report.device_name == "Tesla V100-PCIE-32GB"
    assert report.amp_status == environment.AMP_PASS
    assert report.amp_dtype == "float16"
    assert report.problems == ()
    assert report.ok is True

    lines = "\n".join(report.lines())
    assert "CUDA available: yes" in lines
    assert "AMP pass: pass (float16)" in lines
    assert "Result: PASS" in lines


def test_gpu_without_v100_architecture_is_rejected() -> None:
    report = environment.probe(_fake_torch(architectures=("sm_80", "sm_90")))

    assert report.cuda_available is True
    assert report.ok is False
    assert any("sm_70" in problem for problem in report.problems)
    assert report.amp_status == environment.AMP_PASS


def test_wrong_cuda_build_is_rejected() -> None:
    report = environment.probe(_fake_torch(cuda_version="12.8"))

    assert report.ok is False
    assert any("12.6" in problem for problem in report.problems)


def test_failed_amp_probe_is_reported_as_a_failure() -> None:
    report = environment.probe(_fake_torch(finite=False))

    assert report.amp_status == environment.AMP_FAIL
    assert report.ok is False
    assert any("AMP" in problem for problem in report.problems)


def test_missing_torch_is_reported_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "_import_torch", lambda: None)
    report = environment.probe(None)

    assert report.cuda_available is False
    assert report.amp_status == environment.AMP_NOT_RUN
    assert report.ok is False
    assert any("PyTorch" in problem for problem in report.problems)


def test_check_command_exits_nonzero_and_prints_clear_report_without_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_env.main([], torch_module=_fake_torch(available=False)) == 1

    output = capsys.readouterr().out
    assert "CUDA available: no" in output
    assert "AMP pass: not run" in output
    assert "GPU smoke check: ticket #199 on OCEANO" in output


def test_check_command_exits_zero_with_a_valid_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_env.main([], torch_module=_fake_torch()) == 0

    output = capsys.readouterr().out
    assert "CUDA available: yes" in output
    assert "Result: PASS" in output


# ── Hash verification and the offline wheelhouse ──


def test_sha256_manifest_round_trip(tmp_path: Path) -> None:
    first = tmp_path / "alpha-1.0-py3-none-any.whl"
    second = tmp_path / "beta-2.0-py3-none-any.whl"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    manifest = tmp_path / "SHA256SUMS"

    hashing.write_manifest(manifest, [first, second])

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert hashing.verify_manifest(tmp_path, manifest) == []

    second.write_bytes(b"tampered")
    problems = hashing.verify_manifest(tmp_path, manifest)
    assert len(problems) == 1
    assert "beta-2.0-py3-none-any.whl" in problems[0]


def test_verify_file_accepts_a_matching_digest_and_rejects_a_wrong_one(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"trusted bytes")
    digest = hashing.sha256_file(artifact)

    hashing.verify_file(artifact, digest)

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        hashing.verify_file(artifact, "0" * 64)


def test_fetch_weights_requires_a_manifest_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "weights.sha256"
    manifest.write_text("# only a comment, no pinned digest\n", encoding="utf-8")
    called: list[str] = []

    def downloader(url: str, destination: Path) -> None:
        called.append(url)
        destination.write_bytes(b"unverified")

    with pytest.raises(weights.WeightHashError, match="no weights are pinned"):
        weights.fetch_weights(
            manifest,
            tmp_path / "models",
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )
    assert called == []


def test_fetch_weights_verifies_each_download_and_removes_bad_files(tmp_path: Path) -> None:
    good = b"trusted bytes"
    manifest = tmp_path / "weights.sha256"
    manifest.write_text(
        f"{hashing.sha256_bytes(good)}  model-a.pt\n{'1' * 64}  model-b.pt\n",
        encoding="utf-8",
    )
    destination = tmp_path / "models"
    called: list[str] = []

    def downloader(url: str, path: Path) -> None:
        called.append(url)
        path.write_bytes(good)

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        weights.fetch_weights(
            manifest,
            destination,
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )

    assert (destination / "model-a.pt").read_bytes() == good
    assert not (destination / "model-b.pt").exists()
    assert not (destination / "model-b.pt.part").exists()
    assert called == [
        "https://example.invalid/weights/model-a.pt",
        "https://example.invalid/weights/model-b.pt",
    ]


def test_existing_weight_is_reverified_before_it_is_trusted(tmp_path: Path) -> None:
    payload = b"already on disk"
    manifest = tmp_path / "weights.sha256"
    manifest.write_text(f"{'2' * 64}  cached.pt\n", encoding="utf-8")
    destination = tmp_path / "models"
    destination.mkdir()
    cached = destination / "cached.pt"
    cached.write_bytes(payload)

    def downloader(_url: str, _path: Path) -> None:
        pytest.fail("a cached weight must not be downloaded again")

    with pytest.raises(weights.WeightHashError, match="SHA-256 mismatch"):
        weights.fetch_weights(
            manifest,
            destination,
            base_url="https://example.invalid/weights/",
            downloader=downloader,
        )
    assert not cached.exists()


# ── Wheelhouse plan (no network) ──


def test_wheelhouse_plan_uses_the_cuda_index_and_never_installs() -> None:
    from tools.hpc import build_wheelhouse

    requirements = "\n".join(
        [
            "numpy==1.26.4",
            "torch==2.13.0",
            "torchvision==0.28.0",
            "ultralytics==8.3.40",
        ]
    )
    plan = build_wheelhouse.make_plan(requirements, Path("/wheelhouse"))

    assert plan.default_requirements == ("numpy==1.26.4", "ultralytics==8.3.40")
    assert plan.cuda_requirements == ("torch==2.13.0+cu126", "torchvision==0.28.0+cu126")
    assert build_wheelhouse.CUDA_INDEX in plan.cuda_index_args
    # The builder downloads and builds; installing belongs to the image, so the
    # plan never writes into whichever environment happened to invoke it.
    assert not any(step[:3] == ("uv", "pip", "install") for step in plan.steps)


def test_the_default_pins_drop_the_cuda_runtime_tree() -> None:
    """The lockfile pins CUDA 13; the cu126 wheels declare CUDA 12.6.

    Both in one requirement list is a resolution conflict, so the tree leaves
    the default pins and the framework wheel's metadata selects it instead.
    """
    from tools.hpc import build_wheelhouse

    exported = "\n".join(
        [
            "numpy==1.26.4",
            "torch==2.13.0 ; sys_platform == 'linux'",
            "torchvision==0.28.0 ; sys_platform == 'linux'",
            "cuda-toolkit==13.0.3.0 ; sys_platform == 'linux'",
            "cuda-bindings==13.5.0 ; sys_platform == 'linux'",
            "nvidia-cudnn-cu13==9.20.0.48 ; sys_platform == 'linux'",
            "nvidia-cublas==13.1.1.3 ; sys_platform == 'linux'",
            "triton==3.7.1 ; sys_platform == 'linux'",
        ]
    )
    default, cuda = build_wheelhouse.split_requirements(exported)

    assert default == ("numpy==1.26.4", "triton==3.7.1 ; sys_platform == 'linux'")
    assert cuda == build_wheelhouse.CUDA_PACKAGES


def test_the_cuda_download_resolves_dependencies_against_the_lockfile() -> None:
    """Torch's CUDA tree comes from its own metadata, pinned by constraints."""
    from tools.hpc import build_wheelhouse

    plan = build_wheelhouse.make_plan("numpy==1.26.4\n", Path("/wheelhouse"))
    cuda_step = next(step for step in plan.steps if set(plan.cuda_requirements) <= set(step))

    # Without resolution the wheelhouse would carry torch but not the runtime
    # below it, and the offline install would fail on a node with no index.
    assert "--no-deps" not in cuda_step
    assert cuda_step[cuda_step.index("-c") + 1] == str(plan.wheelhouse / "lock-requirements.txt")
    assert build_wheelhouse.PYPI_INDEX in cuda_step


def test_wheelhouse_plan_filters_torch_before_the_default_index_download(
    tmp_path: Path,
) -> None:
    from tools.hpc import build_wheelhouse

    plan = build_wheelhouse.make_plan("torch==2.13.0\nnumpy==1.26.4\n", tmp_path / "wheelhouse")
    writes = [step[1] for step in plan.steps if step[0] == "write"]
    assert writes[0] == str(plan.wheelhouse / "lock-requirements.txt")

    filtered_at = next(
        index
        for index, step in enumerate(plan.steps)
        if step[0] == "write" and step[1].endswith("lock-requirements.txt")
    )
    download_at = next(
        index for index, step in enumerate(plan.steps) if "pip" in step and "-r" in step
    )
    assert filtered_at < download_at


def test_dry_run_prints_the_offline_install_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tools.hpc import build_wheelhouse

    assert build_wheelhouse.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "cu126" in output
    assert "--no-index" in output
    assert "wheelhouse" in output


def test_the_exclusion_set_is_the_framework_closure_in_the_lockfile() -> None:
    """The dropped pins are exactly what torch and torchvision drag in.

    A prefix filter would also drop a future unrelated ``nvidia-*`` package;
    a stale list would keep a CUDA 13 pin the cu126 wheels contradict. Both are
    this equality: every CUDA-tree package in the lockfile's framework closure
    is dropped, and nothing outside that closure is.
    """
    from tools.hpc import build_wheelhouse

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {entry["name"]: entry for entry in lock["package"]}

    def dependencies(name: str) -> list[str]:
        entry = packages[name]
        edges = [dependency["name"] for dependency in entry.get("dependencies", [])]
        for extra in entry.get("optional-dependencies", {}).values():
            edges += [dependency["name"] for dependency in extra]
        return edges

    seen: set[str] = set()
    pending = ["torch", "torchvision"]
    while pending:
        name = pending.pop()
        if name in seen or name not in packages:
            continue
        seen.add(name)
        pending += dependencies(name)

    in_closure = {name for name in seen if name.startswith(("cuda-", "nvidia-"))}

    assert in_closure == set(build_wheelhouse.CUDA_RUNTIME_PACKAGES), (
        "the exclusion set no longer matches torch's runtime closure in uv.lock"
    )


def test_a_new_cuda_package_keeps_its_lockfile_pin() -> None:
    """Only the named tree is dropped; a prefix is not a policy."""
    from tools.hpc import build_wheelhouse

    exported = "\n".join(
        [
            "cuda-toolkit==13.0.3.0",
            "nvidia-cublas==13.1.1.3",
            "nvidia-something-new==1.0.0",
            "cuda-profiler-sample==2.0.0",
            "numpy==1.26.4",
        ]
    )
    default, _ = build_wheelhouse.split_requirements(exported)

    assert default == (
        "nvidia-something-new==1.0.0",
        "cuda-profiler-sample==2.0.0",
        "numpy==1.26.4",
    )


def test_framework_pins_must_be_exact_builds_of_the_index() -> None:
    """A pin and an index that disagree fail the plan, not a 4 GB download."""
    from tools.hpc import build_wheelhouse

    build_wheelhouse.check_framework_pins(build_wheelhouse.CUDA_PACKAGES)

    with pytest.raises(ValueError, match="not an exact"):
        build_wheelhouse.check_framework_pins(("torch==2.13.0",))
    with pytest.raises(ValueError, match=r"does not belong to .*cu126"):
        build_wheelhouse.check_framework_pins(("torch==2.13.0+cu128",))
    # The index and the tag travel together: pointing the builder at another
    # CUDA build without changing the pins is refused before the download.
    with pytest.raises(ValueError, match=r"does not belong to .*cu128"):
        build_wheelhouse.check_framework_pins(
            build_wheelhouse.CUDA_PACKAGES,
            index="https://download.pytorch.org/whl/cu128",
        )


def test_the_plan_verifies_the_tree_before_it_writes_the_manifest() -> None:
    """A wrong tree must fail before the checksums bless it."""
    from tools.hpc import build_wheelhouse

    plan = build_wheelhouse.make_plan("numpy==1.26.4\n", Path("/wheelhouse"))
    kinds = [step[0] for step in plan.steps]

    verify_at = kinds.index("verify")
    cuda_at = next(
        index for index, step in enumerate(plan.steps) if set(plan.cuda_requirements) <= set(step)
    )
    checksum_at = next(
        index for index, step in enumerate(plan.steps) if step[1].endswith("SHA256SUMS")
    )
    assert cuda_at < verify_at < checksum_at


def test_verify_cuda_runtime_accepts_the_cu126_tree(tmp_path: Path) -> None:
    from tools.hpc import build_wheelhouse

    for name in (
        "torch-2.13.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torchvision-0.28.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl",
        "nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.whl",
        "nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
        "cuda_toolkit-12.6.3-py2.py3-none-any.whl",
        "cuda_bindings-12.9.11-cp311-cp311-manylinux_2_24_x86_64.whl",
        "cuda_pathfinder-1.9.1-py3-none-any.whl",
    ):
        (tmp_path / name).touch()

    assert build_wheelhouse.verify_cuda_runtime(tmp_path) == ()


def test_verify_cuda_runtime_rejects_a_cuda13_tree_or_a_plain_framework(
    tmp_path: Path,
) -> None:
    from tools.hpc import build_wheelhouse

    for name in (
        "torch-2.13.0-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torchvision-0.28.0-cp311-cp311-manylinux_2_28_x86_64.whl",
        "cuda_toolkit-13.0.3.0-py2.py3-none-any.whl",
        "nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl",
        "cuda_bindings-13.5.0-cp311-cp311-manylinux_2_24_x86_64.whl",
    ):
        (tmp_path / name).touch()

    problems = build_wheelhouse.verify_cuda_runtime(tmp_path)

    assert any("expected torch==2.13.0+cu126" in problem for problem in problems)
    assert any("expected torchvision==0.28.0+cu126" in problem for problem in problems)
    assert any("cuda_toolkit-13.0.3.0" in problem for problem in problems)
    assert any("nvidia_cudnn_cu13" in problem for problem in problems)
    assert any("cuda_bindings-13.5.0" in problem for problem in problems)
    assert any("no CUDA 12 runtime wheel" in problem for problem in problems)


def test_the_constrained_resolution_keeps_the_locked_pins(
    tmp_path: Path,
) -> None:
    """Resolve the framework's tree offline, the way the build does.

    A newer runtime and a newer numpy are both on offer. The framework's own
    pin must select the runtime version it declares, while the exported
    lockfile, passed as constraints, keeps every shared pin where it was: that
    is the whole reason the CUDA tree is dropped from the pin list instead of
    being edited to a 12.6 tree that the lockfile does not describe.
    """
    wheelhouse = tmp_path / "wheelhouse"
    _synthetic_wheel(wheelhouse, "torch", "9.9.9+cu126", ("fakecuda==12.6.3", "numpy"))
    _synthetic_wheel(wheelhouse, "torchvision", "9.9.9+cu126", ("torch==9.9.9",))
    _synthetic_wheel(wheelhouse, "fakecuda", "12.6.3")
    _synthetic_wheel(wheelhouse, "fakecuda", "13.0.0")
    _synthetic_wheel(wheelhouse, "numpy", "1.26.4")
    _synthetic_wheel(wheelhouse, "numpy", "2.4.0")

    constraints = tmp_path / "lock-requirements.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")
    pinned = ("torch==9.9.9+cu126", "torchvision==9.9.9+cu126")

    assert _resolve(wheelhouse, constraints, pinned) == {
        "torch==9.9.9+cu126",
        "torchvision==9.9.9+cu126",
        "fakecuda==12.6.3",
        "numpy==1.26.4",
    }
    # Anti-vacuity: without the constraint the resolver picks the newer numpy,
    # so the assertion above is the constraint doing its job, not a default.
    assert "numpy==2.4.0" in _resolve(wheelhouse, None, pinned)


def _synthetic_wheel(
    directory: Path, name: str, version: str, requires: tuple[str, ...] = ()
) -> Path:
    """Write a metadata-only wheel, enough for a resolver to read."""
    directory.mkdir(parents=True, exist_ok=True)
    distribution = f"{name.replace('-', '_')}-{version}"
    filename = directory / f"{distribution}-py3-none-any.whl"
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    metadata += [f"Requires-Dist: {requirement}" for requirement in requires]
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr(f"{distribution}.dist-info/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{distribution}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{distribution}.dist-info/RECORD", "")
    return filename


def _resolve(wheelhouse: Path, constraints: Path | None, requirements: tuple[str, ...]) -> set[str]:
    """What a fresh, environment-free resolution of ``requirements`` selects.

    ``uv pip compile`` resolves without an installed environment, so the answer
    cannot be polluted by whatever torch happens to be in the venv running the
    suite.
    """
    request = wheelhouse / "requirements.txt"
    request.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    command = [
        "uv",
        "pip",
        "compile",
        str(request),
        "--no-config",
        "--no-index",
        "--find-links",
        str(wheelhouse),
    ]
    if constraints is not None:
        command += ["-c", str(constraints)]
    result = subprocess.run(command, cwd=wheelhouse, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


# ── Static evidence for the container definitions ──


def test_container_definitions_pin_the_base_and_the_cuda_build() -> None:
    containerfile = (ROOT / "container" / "Containerfile").read_text(encoding="utf-8")
    definition = (ROOT / "container" / "hpc.def").read_text(encoding="utf-8")

    for text in (containerfile, definition):
        assert "@sha256:" in text
        assert "cu126" in text
        assert "torch==2.13.0+cu126" in text
        assert "--no-index" in text
        # The source commit is stamped into the image; a container has no
        # .git for commit_git() to read.
        assert "SOURCE_SHA" in text
        assert "POLYX_SOURCE_SHA" in text
    assert "Bootstrap: docker" in definition


def test_wheelhouse_plan_builds_the_project_wheel_before_the_checksums() -> None:
    """The node installs the package from a wheel; it never builds it."""
    from tools.hpc import build_wheelhouse

    plan = build_wheelhouse.make_plan("numpy==1.26.4\n", Path("/wheelhouse"))

    build_at = next(index for index, step in enumerate(plan.steps) if step[:2] == ("uv", "build"))
    checksum_at = next(
        index
        for index, step in enumerate(plan.steps)
        if step[0] == "write" and step[1].endswith("SHA256SUMS")
    )
    assert build_at < checksum_at, "the project wheel must be digested with the rest"
    assert "--wheel" in plan.steps[build_at]
    assert "--out-dir" in plan.steps[build_at]


def test_container_definitions_install_the_project_wheel_offline() -> None:
    """A source install needs the network and a build backend; neither is there."""
    containerfile = (ROOT / "container" / "Containerfile").read_text(encoding="utf-8")
    definition = (ROOT / "container" / "hpc.def").read_text(encoding="utf-8")

    # NOTICE is force-included into the wheel, so the build stage must ship it.
    assert "NOTICE ./" in containerfile
    for text in (containerfile, definition):
        assert "micra-*.whl" in text, "the project wheel is not installed"
        assert "--no-index --no-deps /opt/wheelhouse" in text or (
            "--no-index --no-deps /opt/polyx/wheelhouse" in text
        )
        assert "--no-deps ." not in text, "the image still builds the project from source"

    # Apptainer: `uv pip` has no target until the venv is selected.
    post = definition.split("%post", 1)[1].split("%environment", 1)[0]
    assert post.index("export VIRTUAL_ENV=/opt/venv") < post.index("uv pip install")
    assert post.index("uv venv /opt/venv") < post.index("uv pip install")
