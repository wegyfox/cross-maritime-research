"""User-facing release CLI for the qualified CROSS PSP Q1-Q5 profile.

This module is a release wrapper only. It does not alter the frozen scientific
mechanisms or populate the generic D0 default adapter registry.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import as_file, files
from pathlib import Path
import shutil
import sys

from vessel_crossing.cli import main as d0_main
from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.diagnostic.execution import DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY

from .analysis import AnalysisSurfaceError, validate_analysis_scenario, write_analysis_failure, write_analysis_summary

from .reference_cache import default_reference_cache_dir

from .q5_integration import (
    D0_BASE_VERSION,
    Q5_CANDIDATE_VERSION,
    Q5_EXECUTION_IDENTITY_ID,
    Q5IntegrationError,
    build_q5_registry,
    prepare_q5_scenario,
)

RELEASE_VERSION = "1.0.0"
EXPECTED_D0_VERSION = D0_BASE_VERSION
QUALIFIED_INTEGRATION_VERSION = Q5_CANDIDATE_VERSION


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise SystemExit(f"required distribution is not installed: {distribution}") from exc


def _assert_release_environment() -> None:
    d0 = _installed_version("vessel-crossing-simulator")
    psp = _installed_version("cross-psp-two-stream")
    if d0 != EXPECTED_D0_VERSION:
        raise SystemExit(f"unsupported D0 version: expected {EXPECTED_D0_VERSION}, got {d0}")
    if psp != RELEASE_VERSION:
        raise SystemExit(f"unsupported CROSS PSP release: expected {RELEASE_VERSION}, got {psp}")
    if tuple(DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY.adapter_ids) != ():
        raise SystemExit("generic D0 default adapter registry is unexpectedly non-empty")


def _run_scenario(
    scenario_path: Path,
    output: Path,
    run_date: str,
    scope: str,
    title: str | None,
    resume: bool,
    *,
    registry=None,
) -> int:
    _assert_release_environment()
    scenario = load_diagnostic_scenario(scenario_path)
    registry = registry or build_q5_registry(
        scenario, reference_cache_dir=default_reference_cache_dir()
    )
    args = [
        "diagnostic-run",
        "--scenario", str(scenario_path),
        "--output", str(output),
        "--date", run_date,
        "--scope", scope,
    ]
    if title:
        args.extend(["--title", title])
    if resume:
        args.append("--resume")
    return d0_main(args, diagnostic_registry=registry)


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=Path("cross_psp_output"))
    parser.add_argument("--date", dest="run_date", default=date.today().isoformat())
    parser.add_argument("--scope", default="research")
    parser.add_argument("--title")
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cross-psp",
        description="Run the qualified CROSS PSP two-stream Q1-Q5 research profile.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a compatible scenario YAML")
    run.add_argument("scenario", type=Path)
    _add_common_run_arguments(run)

    analyze = sub.add_parser("analyze", help="run a bounded custom demand experiment and write a modeled diagnosis")
    analyze.add_argument("scenario", type=Path)
    _add_common_run_arguments(analyze)

    copy_analysis = sub.add_parser("copy-analysis-template", help="copy the supported custom-analysis scenario template")
    copy_analysis.add_argument("destination", type=Path, nargs="?", default=Path("cross_psp_analysis.yaml"))

    example = sub.add_parser("example", help="run the bundled qualified sparse example")
    _add_common_run_arguments(example)

    copy_example = sub.add_parser("copy-example", help="copy the bundled scenario YAML")
    copy_example.add_argument("destination", type=Path, nargs="?", default=Path("cross_psp_scenario.yaml"))

    sub.add_parser("verify", help="verify exact release/runtime versions and fail-closed defaults")
    sub.add_parser("info", help="print release and qualification identity")

    wizard = sub.add_parser("wizard", help="interactive custom traffic/physics configuration wizard")
    wizard.add_argument("--mode", choices=("traffic", "physics", "load", "robustness"))
    wizard.add_argument("--output", type=Path)
    wizard.add_argument("--config", type=Path)
    return parser


def _handle_verify() -> int:
    _assert_release_environment()
    print(f"CROSS PSP release {RELEASE_VERSION}: PASS")
    print(f"qualified integration: {QUALIFIED_INTEGRATION_VERSION}")
    print(f"D0: {EXPECTED_D0_VERSION}")
    print(f"execution identity: {Q5_EXECUTION_IDENTITY_ID}")
    print("generic D0 defaults: empty/fail-closed")
    return 0


def _handle_info() -> int:
    print(f"release_version={RELEASE_VERSION}")
    print(f"qualified_integration_version={QUALIFIED_INTEGRATION_VERSION}")
    print(f"d0_version={EXPECTED_D0_VERSION}")
    print(f"execution_identity_id={Q5_EXECUTION_IDENTITY_ID}")
    print("field_calibrated=false")
    print("deployment_recommendation=false")
    print("dense_capacity_qualified=false")
    return 0


def _copy_resource(resource, destination: Path) -> int:
    _assert_release_environment()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with as_file(resource) as source:
        shutil.copyfile(source, destination)
    print(destination)
    return 0


def _physics_info(prepared) -> dict[str, object]:
    return {
        "scope_status": prepared.physical.scope_status,
        "physics_identity_sha256": prepared.physical.physics_identity_sha256,
        "reference_mode": "DYNAMIC_SAME_PHYSICS",
        "frozen_a70_reference_value_reused": False,
    }


def _diagnosed_failure(output: Path, validation, error: str) -> int:
    json_path, md_path, terminal = write_analysis_failure(output, validation, error=error)
    print("analysis_status=FAIL_CLOSED_DIAGNOSED")
    print(f"terminal_classification={terminal}")
    print(f"analysis_summary_json={json_path}")
    print(f"analysis_summary_md={md_path}")
    return 0 if terminal != "UNEXPECTED_ERROR" else 2


def _handle_analyze(ns) -> int:
    try:
        validation = validate_analysis_scenario(ns.scenario)
    except AnalysisSurfaceError as exc:
        raise SystemExit("CUSTOM ANALYSIS REJECTED:\n" + str(exc)) from exc

    try:
        scenario_spec = load_diagnostic_scenario(ns.scenario)
        prepared = prepare_q5_scenario(scenario_spec)
        registry = build_q5_registry(
            scenario_spec, reference_cache_dir=default_reference_cache_dir()
        )
    except (Q5IntegrationError, ValueError) as exc:
        return _diagnosed_failure(ns.output, validation, str(exc))

    captured_out = StringIO()
    captured_err = StringIO()
    raised_error = None
    with redirect_stdout(captured_out), redirect_stderr(captured_err):
        try:
            status = _run_scenario(
                ns.scenario,
                ns.output,
                ns.run_date,
                ns.scope,
                ns.title or "CROSS PSP dynamic custom analysis",
                ns.resume,
                registry=registry,
            )
        except Exception as exc:  # profile-owned fail-closed exceptions become evidence
            status = 2
            raised_error = str(exc)
    if captured_out.getvalue():
        print(captured_out.getvalue(), end="")
    if status == 0:
        json_path, md_path = write_analysis_summary(
            ns.output,
            validation,
            physics_info=_physics_info(prepared),
        )
        print("analysis_status=PASS")
        print(f"analysis_summary_json={json_path}")
        print(f"analysis_summary_md={md_path}")
        return 0

    raw_error = captured_err.getvalue().strip()
    error_text = raised_error or raw_error
    if raised_error is None:
        try:
            parsed = json.loads(raw_error)
            error_text = str(parsed.get("error") or raw_error)
        except Exception:
            pass
    result = _diagnosed_failure(ns.output, validation, error_text)
    if result == 2 and raw_error:
        print(raw_error, file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.command == "wizard":
        from .wizard import run_wizard

        def execute(scenario: Path, output: Path) -> int:
            class _NS:
                pass
            run_ns = _NS()
            run_ns.scenario = scenario
            run_ns.output = output
            run_ns.run_date = date.today().isoformat()
            run_ns.scope = "research"
            run_ns.title = "CROSS PSP wizard custom analysis"
            # D0 --resume requires an existing verified PRODUCT_RUN_MANIFEST.json.
            # A first run must be fresh; a later rerun may reuse a validated manifest.
            run_ns.resume = (Path(output) / "PRODUCT_RUN_MANIFEST.json").is_file()
            return _handle_analyze(run_ns)

        return run_wizard(
            mode=ns.mode,
            execute=execute,
            output_dir=ns.output,
            config_path=ns.config,
        )
    if ns.command == "verify":
        return _handle_verify()
    if ns.command == "info":
        return _handle_info()

    example_resource = files("cross_psp_two_stream.data").joinpath("q5_scenario.yaml")
    analysis_resource = files("cross_psp_two_stream.data").joinpath("analysis_template.yaml")
    if ns.command == "copy-analysis-template":
        return _copy_resource(analysis_resource, ns.destination)
    if ns.command == "copy-example":
        return _copy_resource(example_resource, ns.destination)
    if ns.command == "example":
        with as_file(example_resource) as scenario_path:
            return _run_scenario(
                scenario_path,
                ns.output,
                ns.run_date,
                ns.scope,
                ns.title or "CROSS PSP qualified Q1-Q5 example",
                ns.resume,
            )
    if ns.command == "analyze":
        return _handle_analyze(ns)
    if ns.command == "run":
        return _run_scenario(
            ns.scenario,
            ns.output,
            ns.run_date,
            ns.scope,
            ns.title,
            ns.resume,
        )
    raise AssertionError(ns.command)


if __name__ == "__main__":
    raise SystemExit(main())
