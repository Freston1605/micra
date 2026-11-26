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


def _fake_environment(
    root: Path, *, environment_at: float, pyproject_at: float, lock_at: float
) -> Path:
    """A checkout plus the ``site-packages`` of an environment it was synced into."""
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    site = root / ".venv" / "lib" / "python3.12" / "site-packages"
    dist_info = site / "micra-2.0.0.dist-info"
    dist_info.mkdir(parents=True)
    for path, stamp in (
        (root / "pyproject.toml", pyproject_at),
        (root / "uv.lock", lock_at),
        (dist_info, environment_at),
        (site, environment_at),
    ):
        os.utime(path, (stamp, stamp))
    return site


def test_dependency_files_newer_than_the_environment_are_reported(tmp_path, monkeypatch):
    gate = _gate_module()
    site = _fake_environment(
        tmp_path, environment_at=1_000.0, pyproject_at=2_000.0, lock_at=2_000.0
    )
    monkeypatch.setattr(gate, "_site_packages", lambda python: site)

    warning = gate._dependency_drift(tmp_path, Path("/checkout/.venv/bin/python"))

    assert warning is not None
    assert "pyproject.toml and uv.lock" in warning
    assert "uv sync --extra gui" in warning
    assert "/checkout/.venv" in warning


def test_only_the_edited_dependency_file_is_named(tmp_path, monkeypatch):
    gate = _gate_module()
    site = _fake_environment(
        tmp_path, environment_at=2_000.0, pyproject_at=1_000.0, lock_at=3_000.0
    )
    monkeypatch.setattr(gate, "_site_packages", lambda python: site)

    warning = gate._dependency_drift(tmp_path, Path("/checkout/.venv/bin/python"))

    assert warning is not None
    assert "uv.lock changed" in warning
    assert "pyproject.toml" not in warning


def test_a_synced_environment_reports_nothing(tmp_path, monkeypatch):
    gate = _gate_module()
    site = _fake_environment(
        tmp_path, environment_at=3_000.0, pyproject_at=1_000.0, lock_at=1_000.0
    )
    monkeypatch.setattr(gate, "_site_packages", lambda python: site)

    assert gate._dependency_drift(tmp_path, Path("/checkout/.venv/bin/python")) is None


def test_an_environment_without_distributions_is_not_a_finding(tmp_path, monkeypatch):
    """No trace of an install is no opinion, not a stale environment."""
    gate = _gate_module()
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.setattr(gate, "_site_packages", lambda python: site)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert gate._dependency_drift(tmp_path, Path("/checkout/.venv/bin/python")) is None


def test_an_interpreter_that_cannot_report_its_site_packages_is_not_a_finding(
    tmp_path, monkeypatch
):
    gate = _gate_module()
    monkeypatch.setattr(gate, "_site_packages", lambda python: None)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert gate._dependency_drift(tmp_path, Path("/checkout/.venv/bin/python")) is None


def test_the_drift_warning_does_not_stop_the_gate(monkeypatch, capsys):
    """It is a warning: every command still runs and the exit status is unchanged."""
    gate = _gate_module()
    monkeypatch.setattr(gate, "_python", lambda: Path("/checkout/.venv/bin/python"))
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
