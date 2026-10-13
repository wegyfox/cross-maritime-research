"""Command-line entry points for explicit, preflighted protocol execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from .archive import verify_run_package
from .executor import execute_replication_protocol, write_protocol_run_package
from .parallel_executor import execute_replication_protocol_parallel
from .preflight import preflight_protocol_file
from .protocol import load_executable_replication_protocol
from .nextgen.environmental_network_fifo_result_archive import (
    EnvironmentalNetworkFIFOResultArchiveError,
    verify_environmental_network_fifo_result_archive,
)
from .nextgen.environmental_network_fifo_archive_comparison import (
    EnvironmentalNetworkFIFOArchiveComparisonError,
    compare_environmental_network_fifo_result_archive_files,
)
from .reporting.reference_envelope import (
    ReferenceEnvelopeReportingError,
    load_ideal_gap_decomposition,
    write_reference_envelope_artifacts,
)
from .diagnostic import (
    DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
    DiagnosticAdapterRegistry,
    DiagnosticExecutionError,
    DiagnosticResumeError,
    DiagnosticReproductionBundleError,
    DiagnosticRunManifestError,
    DiagnosticScenarioContractError,
    build_diagnostic_plan,
    load_diagnostic_scenario,
    prepare_diagnostic_resume_prefix,
    run_diagnostic_product,
    run_diagnostic_product_with_resume,
    verify_diagnostic_reproduction_bundle,
    write_diagnostic_product_run_manifest,
    write_diagnostic_reproduction_bundle,
    write_diagnostic_resume_evidence,
)


def _report_to_dict(report) -> dict:
    return {
        "execution_protocol_id": report.execution_protocol_id,
        "execution_protocol_fingerprint_sha256": report.execution_protocol_fingerprint_sha256,
        "passed": report.passed,
        "checks": [asdict(check) for check in report.checks],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vessel-crossing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate a frozen execution protocol")
    preflight.add_argument("--protocol", required=True, type=Path)
    preflight.add_argument("--expected-fingerprint")
    preflight.add_argument(
        "--no-require-sidecar",
        action="store_true",
        help="allow preflight without a .fingerprint.sha256 sidecar",
    )

    run = subparsers.add_parser("run", help="preflight then execute a frozen protocol")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--expected-fingerprint")
    run.add_argument("--run-timestamp-utc")
    run.add_argument("--no-require-sidecar", action="store_true")

    parallel = subparsers.add_parser(
        "run-parallel",
        help="preflight then execute/resume a frozen protocol with process workers",
    )
    parallel.add_argument("--protocol", required=True, type=Path)
    parallel.add_argument("--output", required=True, type=Path)
    parallel.add_argument("--checkpoint", required=True, type=Path)
    parallel.add_argument("--workers", type=int, default=None)
    parallel.add_argument("--max-wall-seconds", type=float, default=None)
    parallel.add_argument("--expected-fingerprint")
    parallel.add_argument("--run-timestamp-utc")
    parallel.add_argument("--no-require-sidecar", action="store_true")

    verify = subparsers.add_parser("verify-package", help="verify an archived run package")
    verify.add_argument("--package", required=True, type=Path)

    verify_fifo = subparsers.add_parser(
        "verify-fifo-result",
        help="verify one checksum-backed A152/A153 FIFO result archive",
    )
    verify_fifo.add_argument("--result", required=True, type=Path)

    compare_fifo = subparsers.add_parser(
        "compare-fifo-results",
        help="compare two checksum-backed FIFO result archives without executing them",
    )
    compare_fifo.add_argument("--left", required=True, type=Path)
    compare_fifo.add_argument("--right", required=True, type=Path)
    compare_fifo.add_argument("--max-difference-paths", type=int, default=128)
    compare_fifo.add_argument("--format", choices=("json", "text"), default="json")

    render_reference = subparsers.add_parser(
        "render-reference-report",
        help="validate an ideal-gap decomposition and render canonical JSON/HTML report artifacts",
    )
    render_reference.add_argument("--input", required=True, type=Path)
    render_reference.add_argument("--output", required=True, type=Path)
    render_reference.add_argument("--title")

    diagnostic_validate = subparsers.add_parser(
        "diagnostic-validate",
        help="validate a D0 YAML/JSON diagnostic scenario contract",
    )
    diagnostic_validate.add_argument("--scenario", required=True, type=Path)

    diagnostic_plan = subparsers.add_parser(
        "diagnostic-plan",
        help="validate a D0 scenario and build the canonical B1-B6 intervention plan",
    )
    diagnostic_plan.add_argument("--scenario", required=True, type=Path)

    diagnostic_run = subparsers.add_parser(
        "diagnostic-run",
        help="execute one D0 scenario through explicit runtime bindings and render a report",
    )
    diagnostic_run.add_argument("--scenario", required=True, type=Path)
    diagnostic_run.add_argument("--output", required=True, type=Path)
    diagnostic_run.add_argument("--date", required=True)
    diagnostic_run.add_argument("--scope", required=True)
    diagnostic_run.add_argument("--title")
    diagnostic_run.add_argument(
        "--resume",
        action="store_true",
        help="reuse a verified contiguous prefix from PRODUCT_RUN_MANIFEST.json/RUN_EVIDENCE",
    )

    verify_diagnostic_bundle = subparsers.add_parser(
        "verify-diagnostic-bundle",
        help="verify one D0 reproduction bundle without scientific re-execution",
    )
    verify_diagnostic_bundle.add_argument("--bundle", required=True, type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    diagnostic_registry: DiagnosticAdapterRegistry = DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
) -> int:
    args = build_parser().parse_args(argv)

    if args.command in {"diagnostic-validate", "diagnostic-plan"}:
        try:
            scenario = load_diagnostic_scenario(args.scenario)
            if args.command == "diagnostic-validate":
                payload = {
                    "command": args.command,
                    "passed": True,
                    "scenario_id": scenario.scenario_id,
                    "scenario_sha256": scenario.scenario_sha256,
                    "model_adapters": dict(scenario.model_adapters),
                    "requested_interventions": list(scenario.requested_interventions),
                    "claim_scope": dict(scenario.claim_scope),
                }
            else:
                plan = build_diagnostic_plan(scenario)
                payload = plan.to_json_object()
                payload["command"] = args.command
                payload["passed"] = True
                payload["plan_sha256"] = plan.plan_sha256
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        except (DiagnosticScenarioContractError, OSError, ValueError, TypeError) as exc:
            print(
                json.dumps(
                    {"command": args.command, "passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if args.command == "verify-diagnostic-bundle":
        verification = verify_diagnostic_reproduction_bundle(args.bundle)
        payload = {
            "command": args.command,
            "passed": verification.passed,
            "bundle_sha256": verification.bundle_sha256,
            "manifest_sha256": verification.manifest_sha256,
            "errors": list(verification.errors),
        }
        stream = sys.stdout if verification.passed else sys.stderr
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=stream)
        return 0 if verification.passed else 2

    if args.command == "diagnostic-run":
        try:
            scenario = load_diagnostic_scenario(args.scenario)
            prefix = None
            if args.resume:
                prefix = prepare_diagnostic_resume_prefix(
                    scenario,
                    registry=diagnostic_registry,
                    manifest_path=args.output / "PRODUCT_RUN_MANIFEST.json",
                    evidence_directory=args.output,
                )
                result = run_diagnostic_product_with_resume(
                    scenario,
                    registry=diagnostic_registry,
                    prefix=prefix,
                    output_directory=args.output,
                    date=args.date,
                    scope=args.scope,
                    title=args.title,
                )
            else:
                result = run_diagnostic_product(
                    scenario,
                    registry=diagnostic_registry,
                    output_directory=args.output,
                    date=args.date,
                    scope=args.scope,
                    title=args.title,
                )

            run_manifest_payload = None
            evidence_payload = None
            reproduction_bundle_payload = None
            if "execution_identity_sha256" in result.execution.study.provenance:
                evidence_artifacts = write_diagnostic_resume_evidence(
                    result.execution.study,
                    args.output,
                )
                run_manifest = write_diagnostic_product_run_manifest(
                    result,
                    scenario,
                    args.output,
                    date=args.date,
                    scope=args.scope,
                )
                evidence_payload = {
                    "directory": evidence_artifacts.directory,
                    "file_count": len(evidence_artifacts.files),
                }
                run_manifest_payload = {
                    "path": run_manifest.path,
                    "sha256": run_manifest.manifest_sha256,
                }
                reproduction_bundle = write_diagnostic_reproduction_bundle(
                    result,
                    scenario,
                    args.output,
                    date=args.date,
                    scope=args.scope,
                )
                reproduction_bundle_payload = {
                    "path": reproduction_bundle.path,
                    "sha256": reproduction_bundle.bundle_sha256,
                    "manifest_sha256": reproduction_bundle.manifest_sha256,
                    "file_count": reproduction_bundle.file_count,
                }

            classification = result.execution.classification
            output_payload = {
                "command": args.command,
                "passed": True,
                "scenario_id": scenario.scenario_id,
                "scenario_sha256": scenario.scenario_sha256,
                "plan_sha256": result.execution.study.provenance.get("plan_sha256"),
                "runtime_adapter": result.execution.study.provenance.get("runtime_adapter"),
                "classification": classification.classification,
                "selected_intervention": classification.intervention_id,
                "output": result.artifacts.output_directory,
                "diagnostics": {
                    "path": result.artifacts.diagnostics_path,
                    "sha256": result.artifacts.diagnostics_sha256,
                },
                "report_html": {
                    "path": result.artifacts.report_html_path,
                    "sha256": result.artifacts.report_html_sha256,
                },
                "manifest": {
                    "path": result.artifacts.manifest_path,
                    "sha256": result.artifacts.manifest_sha256,
                },
                "decision_view": {
                    "json_path": result.decision_view.json_path,
                    "json_sha256": result.decision_view.json_sha256,
                    "html_path": result.decision_view.html_path,
                    "html_sha256": result.decision_view.html_sha256,
                },
                "run_manifest": run_manifest_payload,
                "run_evidence": evidence_payload,
                "reproduction_bundle": reproduction_bundle_payload,
                "resume": {
                    "requested": bool(args.resume),
                    "verified_through": (
                        prefix.verified_through.value
                        if prefix is not None and prefix.verified_through is not None
                        else None
                    ),
                    "complete_prefix": prefix.complete if prefix is not None else False,
                    "stop_reason": prefix.stop_reason if prefix is not None else None,
                },
            }
            print(
                json.dumps(
                    output_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        except (
            DiagnosticExecutionError,
            DiagnosticResumeError,
            DiagnosticReproductionBundleError,
            DiagnosticRunManifestError,
            DiagnosticScenarioContractError,
            ReferenceEnvelopeReportingError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            print(
                json.dumps(
                    {"command": args.command, "passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if args.command == "verify-fifo-result":
        try:
            archive = verify_environmental_network_fifo_result_archive(
                args.result, require_checksum=True
            )
            document = archive.to_json_object()
            payload = {
                "passed": True,
                "kind": archive.kind.value,
                "format": archive.format,
                "schema_version": archive.schema_version,
                "result_sha256": archive.result_sha256,
                "config_sha256": document.get("config_sha256"),
                "run_id": document.get("run_id"),
                "pair_id": document.get("pair_id"),
                "checksum_verified": archive.checksum_verified,
                "size_bytes": archive.size_bytes,
                "source_path": archive.source_path,
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        except (EnvironmentalNetworkFIFOResultArchiveError, OSError, ValueError) as exc:
            print(
                json.dumps(
                    {"command": args.command, "passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if args.command == "compare-fifo-results":
        try:
            comparison = compare_environmental_network_fifo_result_archive_files(
                args.left,
                args.right,
                max_difference_paths=args.max_difference_paths,
                require_checksum=True,
            )
            if args.format == "text":
                sys.stdout.write(comparison.to_text())
            else:
                sys.stdout.write(comparison.to_json_bytes().decode("utf-8") + "\n")
            return 0
        except (
            EnvironmentalNetworkFIFOResultArchiveError,
            EnvironmentalNetworkFIFOArchiveComparisonError,
            OSError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {"command": args.command, "passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if args.command == "render-reference-report":
        try:
            decomposition = load_ideal_gap_decomposition(args.input)
            source_file_sha256 = sha256(args.input.read_bytes()).hexdigest()
            artifacts = write_reference_envelope_artifacts(
                decomposition,
                args.output,
                title=args.title,
                source_file_sha256=source_file_sha256,
            )
            print(
                json.dumps(
                    {
                        "command": args.command,
                        "passed": True,
                        "scenario_ids": list(decomposition.scenario_ids),
                        "source_decomposition_canonical_sha256": decomposition.result_sha256,
                        "source_file_sha256": source_file_sha256,
                        "output": artifacts.output_directory,
                        "diagnostics": {
                            "path": artifacts.diagnostics_path,
                            "sha256": artifacts.diagnostics_sha256,
                        },
                        "report_html": {
                            "path": artifacts.report_html_path,
                            "sha256": artifacts.report_html_sha256,
                        },
                        "manifest": {
                            "path": artifacts.manifest_path,
                            "sha256": artifacts.manifest_sha256,
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        except (ReferenceEnvelopeReportingError, OSError, ValueError, TypeError) as exc:
            print(
                json.dumps(
                    {"command": args.command, "passed": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if args.command == "verify-package":
        result = verify_run_package(args.package)
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "checked_files": [asdict(item) for item in result.checked_files],
                    "errors": list(result.errors),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if result.passed else 2

    report = preflight_protocol_file(
        args.protocol,
        expected_fingerprint=args.expected_fingerprint,
        require_fingerprint_sidecar=not args.no_require_sidecar,
    )
    if args.command == "preflight":
        print(json.dumps(_report_to_dict(report), ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if report.passed else 2

    if not report.passed:
        print(json.dumps(_report_to_dict(report), ensure_ascii=False, sort_keys=True, indent=2))
        return 2

    protocol = load_executable_replication_protocol(args.protocol)
    if args.command == "run-parallel":
        progress = execute_replication_protocol_parallel(
            protocol,
            workers=args.workers,
            checkpoint_directory=args.checkpoint,
            max_wall_seconds=args.max_wall_seconds,
            run_timestamp_utc=args.run_timestamp_utc,
        )
        if not progress.completed:
            payload = {
                "preflight": _report_to_dict(report),
                "completed": False,
                "checkpoint": progress.checkpoint_path,
                "completed_replications_by_point": dict(
                    progress.completed_replications_by_point
                ),
                "fully_completed_replications": progress.fully_completed_replications,
                "next_target_replications": progress.next_target_replications,
                "batch_end_counts": list(progress.batch_end_counts),
                "tasks_completed_this_call": progress.tasks_completed_this_call,
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            return 4
        result = progress.result
        assert result is not None
    else:
        result = execute_replication_protocol(
            protocol,
            run_timestamp_utc=args.run_timestamp_utc,
        )

    write_protocol_run_package(result, args.output)
    verification = verify_run_package(args.output)
    payload = {
        "preflight": _report_to_dict(report),
        "run_id": result.manifest.run_id,
        "stopping_reason": result.manifest.stopping_reason.value,
        "final_replications": result.manifest.final_replications,
        "output": str(args.output),
        "package_verified": verification.passed,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if verification.passed else 3


if __name__ == "__main__":
    sys.exit(main())
