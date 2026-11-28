"""The local gate forwards the selection and parallelises only on request.

The suite is the slow part of the gate, so an explicit ``--parallel`` opt-in
can use every core. Serial stays the default because CI and ordering-sensitive
tests need a reproducible reference; parallelism must never be implicit.

The gate also reports dependency drift: an environment synced before
``pyproject.toml`` or ``uv.lock`` changed may not be the stack the branch
develops against, and ``uv lock --check`` cannot see that.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "tools" / "check.py"

# The temporary test checks the two facts the gate controls: which ``src`` the
# imports resolve to, and whether xdist workers exist only when requested.
_TEMP_TEST = """\
import os
from pathlib import Path

import polyx


def test_imports_resolve_to_the_gate_worktree():
    source = Path(os.environ["POLYX_EXPECTED_SRC"]).resolve()
    assert Path(polyx.__file__).resolve().is_relative_to(source)


def test_parallelism_matches_the_request():
    expected = os.environ.get("POLYX_EXPECT_PARALLEL") == "1"
    assert ("PYTEST_XDIST_WORKER" in os.environ) is expected
"""


def _run_gate(tmp_path: Path, *args: str, parallel: bool):
    tests = tmp_path / "test_gate_seam.py"
    tests.write_text(_TEMP_TEST, encoding="utf-8")
    env = os.environ.copy()
    # An enclosing xdist run exports these; the inner run must decide for itself.
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_XDIST_WORKER_COUNT", None)
    env["POLYX_EXPECTED_SRC"] = str(_ROOT / "src")
    env["POLYX_EXPECT_PARALLEL"] = "1" if parallel else "0"
    command = [
        sys.executable,
        str(_GATE),
        str(tests),
        "-vv",
        "-p",
        "no:cacheprovider",
        *args,
    ]
    return subprocess.run(
        command,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _assert_only_selection_ran(result: subprocess.CompletedProcess[str], expected: int) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASSED") == expected
    assert "test_gate_seam.py" in result.stdout


def test_selection_runs_serially_by_default(tmp_path):
    result = _run_gate(tmp_path, parallel=False)

    _assert_only_selection_ran(result, expected=2)


def test_parallel_is_an_explicit_opt_in(tmp_path):
    result = _run_gate(tmp_path, "--parallel", parallel=True)

    _assert_only_selection_ran(result, expected=2)


def test_parallel_accepts_an_explicit_worker_count(tmp_path):
    result = _run_gate(tmp_path, "--parallel=1", parallel=True)

    _assert_only_selection_ran(result, expected=2)
    assert "-n 1" in result.stdout
    assert "-n auto" not in result.stdout


def _gate_module():
    spec = importlib.util.spec_from_file_location("polyx_check_gate", _GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gate_checks_the_lockfile_and_the_palette(monkeypatch):
    gate = _gate_module()
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    commands = gate._commands(Path("/python"), [], [])

    palette = str(gate.ROOT / "tools" / "check_palette_fallbacks.py")
    assert commands[0] == ["/usr/bin/uv", "lock", "--check"]
    assert any(palette in command for command in commands)
    assert any("check_i18n.py" in part for command in commands for part in command)


def test_a_missing_uv_only_drops_the_lockfile_check(monkeypatch):
    gate = _gate_module()
    monkeypatch.setattr(gate.shutil, "which", lambda name: None)

    commands = gate._commands(Path("/python"), [], [])

    assert all(Path(command[0]).name != "uv" for command in commands)
    assert any("check_i18n.py" in part for command in commands for part in command)


def test_incomplete_worktree_interpreter_is_skipped(tmp_path):
    gate = _gate_module()
    missing = tmp_path / "missing-python"
    incomplete = tmp_path / "incomplete-python"
    complete = tmp_path / "complete-python"
    incomplete.write_text("", encoding="utf-8")
    complete.write_text("", encoding="utf-8")

    chosen, skipped = gate._select_interpreter(
        [missing, incomplete, complete], lambda path: path == complete
    )

    assert chosen == complete
    assert skipped == [incomplete]


def test_static_skips_the_suite(monkeypatch):
    gate = _gate_module()
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    commands = gate._commands(Path("/python"), [], [], static=True)

    assert all("pytest" not in command for command in commands)
    assert any("check_i18n.py" in part for command in commands for part in command)
    assert any("ruff" in command for command in commands)
    assert any("ty" in command for command in commands)


def test_static_rejects_pytest_arguments():
    result = subprocess.run(
        [sys.executable, str(_GATE), "--static", "tests/test_labels.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--static" in result.stderr


def _interpreter(checkout: Path) -> Path:
    """The interpreter inside a checkout's ``.venv``, with the OS's layout."""
    if os.name == "nt":
        return checkout / ".venv" / "Scripts" / "python.exe"
    return checkout / ".venv" / "bin" / "python"


def _fake_environment(
    root: Path, *, environment_at: float, pyproject_at: float, lock_at: float
) -> tuple[Path, Path]:
    """A checkout plus the environment it was synced into.

    Returns the ``sys.prefix`` and ``site-packages`` a real probe would report,
    so ``_dependency_drift`` is exercised without a subprocess.
    """
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    prefix = root / ".venv"
    site = prefix / "lib" / "python3.12" / "site-packages"
    dist_info = site / "micra-2.0.0.dist-info"
    dist_info.mkdir(parents=True)
    for path, stamp in (
        (root / "pyproject.toml", pyproject_at),
        (root / "uv.lock", lock_at),
        (dist_info, environment_at),
        (site, environment_at),
    ):
        os.utime(path, (stamp, stamp))
    return prefix, site


def test_dependency_files_newer_than_the_environment_are_reported(tmp_path, monkeypatch):
    gate = _gate_module()
    prefix, site = _fake_environment(
        tmp_path, environment_at=1_000.0, pyproject_at=2_000.0, lock_at=2_000.0
    )
    monkeypatch.setattr(gate, "_environment", lambda python: (prefix, site))

    warning = gate._dependency_drift(tmp_path, _interpreter(tmp_path))

    assert warning is not None
    assert "pyproject.toml and uv.lock" in warning
    assert "uv sync --extra gui" in warning
    assert str(prefix) in warning


def test_only_the_edited_dependency_file_is_named(tmp_path, monkeypatch):
    gate = _gate_module()
    prefix, site = _fake_environment(
        tmp_path, environment_at=2_000.0, pyproject_at=1_000.0, lock_at=3_000.0
    )
    monkeypatch.setattr(gate, "_environment", lambda python: (prefix, site))

    warning = gate._dependency_drift(tmp_path, _interpreter(tmp_path))

    assert warning is not None
    assert "uv.lock changed" in warning
    assert "pyproject.toml" not in warning


def test_a_synced_environment_reports_nothing(tmp_path, monkeypatch):
    gate = _gate_module()
    prefix, site = _fake_environment(
        tmp_path, environment_at=3_000.0, pyproject_at=1_000.0, lock_at=1_000.0
    )
    monkeypatch.setattr(gate, "_environment", lambda python: (prefix, site))

    assert gate._dependency_drift(tmp_path, _interpreter(tmp_path)) is None


def test_an_environment_without_distributions_is_not_a_finding(tmp_path, monkeypatch):
    """No trace of an install is no opinion, not a stale environment."""
    gate = _gate_module()
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(gate, "_environment", lambda python: (tmp_path / ".venv", site))
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert gate._dependency_drift(tmp_path, _interpreter(tmp_path)) is None


def test_an_interpreter_that_cannot_report_its_environment_is_not_a_finding(tmp_path, monkeypatch):
    gate = _gate_module()
    monkeypatch.setattr(gate, "_environment", lambda python: None)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert gate._dependency_drift(tmp_path, _interpreter(tmp_path)) is None


def test_the_real_probe_reads_a_created_environment(tmp_path):
    """The probe runs an interpreter, not a monkeypatched path.

    A throwaway venv makes ``sys.prefix`` and ``purelib`` answer something the
    test controls, so the subprocess boundary is exercised end to end.
    """
    import venv

    gate = _gate_module()
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    found = gate._environment(python)

    assert found is not None
    prefix, site = found
    assert prefix.resolve() == environment.resolve()
    assert site.resolve().is_relative_to(environment.resolve())

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (checkout / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    dist_info = site / "micra-2.0.0.dist-info"
    dist_info.mkdir()
    for path in (checkout / "pyproject.toml", checkout / "uv.lock"):
        os.utime(path, (2_000.0, 2_000.0))
    for path in (site, dist_info):
        os.utime(path, (1_000.0, 1_000.0))

    warning = gate._dependency_drift(checkout, python)

    assert warning is not None
    assert str(environment) in warning
    assert "uv sync --extra gui" in warning


def test_the_drift_warning_does_not_stop_the_gate(tmp_path, monkeypatch, capsys):
    """It is a warning: every command still runs and the exit status is unchanged."""
    gate = _gate_module()
    monkeypatch.setattr(gate, "_python", lambda: _interpreter(tmp_path))
    monkeypatch.setattr(
        gate,
        "_dependency_drift",
        lambda root, python: "uv.lock changed: run `uv sync --extra gui`",
    )
    monkeypatch.setattr(gate, "_commands", lambda *args, **kwargs: [["check"]])
    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    ran: list[list[str]] = []

    def record(command, env):
        ran.append(command)
        return 0

    monkeypatch.setattr(gate, "_run", record)

    assert gate.main([]) == 0
    assert ran == [["check"]]
    assert "uv sync --extra gui" in capsys.readouterr().err
