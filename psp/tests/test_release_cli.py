from __future__ import annotations

from cross_psp_two_stream.cli import (
    EXPECTED_D0_VERSION,
    QUALIFIED_INTEGRATION_VERSION,
    RELEASE_VERSION,
    build_parser,
)


def test_release_identity_is_explicit() -> None:
    assert RELEASE_VERSION == "1.0.0"
    assert QUALIFIED_INTEGRATION_VERSION == "0.1.0a8"
    assert EXPECTED_D0_VERSION == "1.1.0a36"


def test_cli_surface_has_release_commands() -> None:
    parser = build_parser()
    for argv in (
        ["verify"],
        ["info"],
        ["example"],
        ["copy-example"],
        ["run", "scenario.yaml"],
        ["analyze", "scenario.yaml"],
        ["copy-analysis-template"],
    ):
        ns = parser.parse_args(argv)
        assert ns.command


def test_wizard_cli_first_run_does_not_force_resume(tmp_path, monkeypatch) -> None:
    import cross_psp_two_stream.cli as cli

    scenario = tmp_path / "scenario.yaml"
    output = tmp_path / "output"
    captured = []

    def fake_wizard(**kwargs):
        execute = kwargs["execute"]
        scenario.write_text("placeholder", encoding="utf-8")
        return execute(scenario, output)

    def fake_handle(ns):
        captured.append(bool(ns.resume))
        return 0

    monkeypatch.setattr("cross_psp_two_stream.wizard.run_wizard", fake_wizard)
    monkeypatch.setattr(cli, "_handle_analyze", fake_handle)
    assert cli.main(["wizard", "--mode", "traffic", "--output", str(output)]) == 0
    assert captured == [False]

    output.mkdir(parents=True)
    (output / "PRODUCT_RUN_MANIFEST.json").write_text("{}", encoding="utf-8")
    captured.clear()
    assert cli.main(["wizard", "--mode", "traffic", "--output", str(output)]) == 0
    assert captured == [True]


def test_release_identity_matches_project_and_launcher() -> None:
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == RELEASE_VERSION

    launcher = (root / "release_shell" / "START_CROSS.ps1").read_text(encoding="utf-8")
    match = re.search(r'\$requiredVersion\s*=\s*"([^"]+)"', launcher)
    assert match is not None
    assert match.group(1) == RELEASE_VERSION
