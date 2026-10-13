"""Interactive terminal wizard for RC2 custom traffic and custom physics.

The wizard is deliberately a thin UX layer.  It generates a normal diagnostic YAML,
preflights it through the same public validation/integration path as ``cross-psp analyze``,
and then delegates execution back to the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import sys
from threading import Event, Thread
from time import monotonic
from typing import Any, Callable, Mapping

from .wizard_config import (
    StreamPhysicsInput,
    TrafficInput,
    WizardPreflight,
    build_physics_scenario,
    build_traffic_scenario,
    preflight_scenario,
    write_scenario,
)


class WizardCancelled(Exception):
    """Internal control flow for an explicit user cancellation."""


class WizardBack(Exception):
    """Internal control flow for returning to the release launcher/menu."""


@dataclass(slots=True)
class ConsoleIO:
    read: Callable[[str], str] = input
    write: Callable[[str], None] = print


def _fmt_minutes(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} sec"
    return f"{seconds / 60.0:.1f} min"


def _prompt(io: ConsoleIO, label: str, default: str, parse: Callable[[str], object]) -> object:
    while True:
        raw = io.read(f"{label} [{default}] > ").strip()
        if raw.lower() in {"q", "quit", "cancel", "/cancel"}:
            raise WizardCancelled
        if raw.lower() in {"b", "back", "/back"}:
            raise WizardBack
        if raw == "":
            raw = default
        try:
            return parse(raw)
        except (TypeError, ValueError) as exc:
            io.write(f"Invalid value: {exc}")


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise ValueError("must be positive")
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise ValueError("must be between 0 and 1")
    return value


def _int_at_least_2(text: str) -> int:
    value = int(text)
    if value < 2:
        raise ValueError("must be at least 2")
    return value


def _traffic(io: ConsoleIO) -> TrafficInput:
    io.write("Traffic/statistical inputs. Press Enter to keep the shown default; type 'back' or 'cancel'.")
    return TrafficInput(
        alpha_rate_per_hour=float(_prompt(io, "Alpha arrivals, vessels/hour", "6.0", _positive_float)),
        beta_rate_per_hour=float(_prompt(io, "Beta arrivals, vessels/hour", "6.0", _positive_float)),
        seed=int(_prompt(io, "Random seed", "359", int)),
        traffic_horizon_min=float(_prompt(io, "Traffic-generation horizon, min", "15.0", _positive_float)),
        end_time_min=float(_prompt(io, "Simulation end time, min", "55.0", _positive_float)),
        replications=int(_prompt(io, "Paired replications", "2", _int_at_least_2)),
        confidence_level=float(_prompt(io, "Confidence level", "0.95", _fraction)),
        seed_namespace=str(_prompt(io, "Seed namespace", "cross-psp-q5-v1", str)),
    )


def _domain(io: ConsoleIO, prefix: str) -> dict[str, object]:
    while True:
        kind = str(_prompt(io, f"{prefix} safety domain: 1=circular, 2=four-quadrant", "1", str))
        if kind == "1":
            radius = float(_prompt(io, f"{prefix} circular radius, NM", "0.10", _positive_float))
            return {"type": "circular_nm", "radius_nm": radius}
        if kind == "2":
            return {
                "type": "four_quadrant_m",
                "ahead_m": float(_prompt(io, f"{prefix} domain ahead, m", "185.2", _positive_float)),
                "astern_m": float(_prompt(io, f"{prefix} domain astern, m", "185.2", _positive_float)),
                "port_m": float(_prompt(io, f"{prefix} domain port, m", "185.2", _positive_float)),
                "starboard_m": float(_prompt(io, f"{prefix} domain starboard, m", "185.2", _positive_float)),
            }
        io.write("Invalid value: choose 1 or 2.")


def _stream_physics(io: ConsoleIO, stream_id: str, heading_default: float) -> StreamPhysicsInput:
    title = stream_id.capitalize()
    io.write(f"\n{title} physical parameters")
    return StreamPhysicsInput(
        stream_id=stream_id,
        source_to_center_nm=float(_prompt(io, f"{title} source-to-center distance, NM", "1.5", _positive_float)),
        center_to_exit_nm=float(_prompt(io, f"{title} center-to-exit distance, NM", "1.5", _positive_float)),
        heading_deg=float(_prompt(io, f"{title} route heading, deg", str(heading_default), float)),
        cruise_speed_knots=float(_prompt(io, f"{title} cruise speed, knots", "6.0", _positive_float)),
        maximum_speed_knots=float(_prompt(io, f"{title} maximum speed, knots", "10.0", _positive_float)),
        length_m=float(_prompt(io, f"{title} vessel length, m", "120.0", _positive_float)),
        beam_m=float(_prompt(io, f"{title} vessel beam, m", "20.0", _positive_float)),
        max_acceleration_knots_per_min=float(_prompt(io, f"{title} max acceleration, knots/min", "2.0", _positive_float)),
        max_deceleration_knots_per_min=float(_prompt(io, f"{title} max deceleration, knots/min", "2.0", _positive_float)),
        max_turn_rate_deg_per_min=float(_prompt(io, f"{title} max turn rate, deg/min", "30.0", _positive_float)),
        safety_domain=_domain(io, title),
    )


def _summary(io: ConsoleIO, path: Path, preflight: WizardPreflight) -> None:
    lo, hi = preflight.estimated_wall_clock_seconds
    io.write("\nPreflight PASS")
    io.write(f"  config: {path}")
    io.write(f"  scientific scope: {preflight.scope_status}")
    io.write(f"  physics identity: {preflight.physics_identity_sha256}")
    v = preflight.validation
    io.write(f"  arrivals: {dict(v.rate_per_hour_by_stream)} vessels/hour")
    io.write(f"  seed / traffic horizon / end: {v.seed} / {v.traffic_horizon_min} min / {v.end_time} min")
    io.write(f"  replications / confidence: {v.replications} / {v.confidence_level}")
    io.write(f"  physical fields changed: {len(v.physical_changed_paths)}")
    io.write(f"  diagnostic simulation runs: {preflight.diagnostic_replication_runs}")
    io.write(f"  dynamic B6 reference runs needed now: {preflight.dynamic_reference_runs_if_cache_miss}")
    io.write(f"  verified reference cache hit: {str(preflight.verified_reference_cache_hit).lower()}")
    io.write(f"  measured-calibration ETA range: {_fmt_minutes(lo)} - {_fmt_minutes(hi)}")
    io.write(f"  ETA basis: {preflight.estimate_basis}")
    io.write("  output: REPORT.html, ANALYSIS_SUMMARY.json/.md, decision/evidence/reproduction bundle")
    io.write("  claim boundary: modeled research diagnosis; not field calibrated or deployment advice")


def _yes(io: ConsoleIO, prompt: str, default: bool = False) -> bool:
    token = "Y/n" if default else "y/N"
    while True:
        raw = io.read(f"{prompt} [{token}] > ").strip().lower()
        if raw in {"q", "quit", "cancel"}:
            raise WizardCancelled
        if raw in {"b", "back"}:
            raise WizardBack
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        io.write("Please answer Y or N, or type back/cancel.")




def _execute_with_heartbeat(
    io: ConsoleIO,
    execute: Callable[[Path, Path], int],
    scenario: Path,
    output: Path,
    preflight: WizardPreflight,
) -> int:
    label = (
        "diagnostic B0-B6 (verified dynamic-reference cache hit)"
        if preflight.verified_reference_cache_hit
        else "fresh dynamic B6 reference + diagnostic B0-B6"
    )
    started = monotonic()
    stop = Event()
    resume_manifest = (Path(output) / "PRODUCT_RUN_MANIFEST.json").is_file()
    restart_mode = "verified-manifest-resume" if resume_manifest else "fresh-diagnostic"
    io.write(
        f"Execution progress: 0/1 campaign; current={label}; restart_mode={restart_mode}; "
        "config_saved=true; completed_reference_cache_reusable=true"
    )

    def heartbeat() -> None:
        while not stop.wait(30.0):
            elapsed = monotonic() - started
            io.write(
                f"Execution progress: 0/1 campaign; current={label}; "
                f"elapsed={_fmt_minutes(elapsed)}; restart_mode={restart_mode}; "
                "config_saved=true; completed_reference_cache_reusable=true"
            )

    thread = Thread(target=heartbeat, name="cross-wizard-heartbeat", daemon=True)
    thread.start()
    try:
        status = int(execute(scenario, output))
    finally:
        stop.set()
        thread.join(timeout=1.0)
    elapsed = monotonic() - started
    io.write(f"Execution progress: 1/1 campaign; elapsed={_fmt_minutes(elapsed)}; status={status}")
    return status




def _default_open_result(path: Path) -> None:
    """Open both the result folder and primary result on Windows.

    The folder is opened first so the primary result can remain the foreground window.
    Non-Windows qualification hosts intentionally do nothing.
    """
    if sys.platform == "win32":
        os.startfile(str(path.parent))  # type: ignore[attr-defined]
        os.startfile(str(path))  # type: ignore[attr-defined]

def _default_config_path(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("cross_custom_configs") / f"cross_{mode}_{stamp}.yaml"


def _new_scenario(io: ConsoleIO, mode: str) -> dict[str, object]:
    traffic = _traffic(io)
    if mode == "traffic":
        return build_traffic_scenario(traffic)
    alpha = _stream_physics(io, "alpha", 0.0)
    beta = _stream_physics(io, "beta", 270.0)
    share = float(_prompt(io, "B6 target share for alpha (beta gets remainder)", "0.5", _fraction))
    return build_physics_scenario(traffic, alpha, beta, alpha_b6_share=share)


def run_wizard(
    *,
    mode: str | None = None,
    io: ConsoleIO | None = None,
    execute: Callable[[Path, Path], int] | None = None,
    output_dir: Path | None = None,
    open_result: Callable[[Path], None] | None = None,
    config_path: Path | None = None,
    robustness_execute_unit: Callable[[Any], Mapping[str, Any]] | None = None,
    robustness_execution_code_identity_sha256: str | None = None,
) -> int:
    """Run the interactive custom-analysis wizard.

    ``execute`` is injected by the CLI so the wizard remains independent of CLI parsing and
    scientific execution. Tests can therefore qualify UX/config generation without running
    expensive simulations.
    """
    io = io or ConsoleIO()
    try:
        selected = mode
        if selected is None:
            io.write("CROSS custom analysis wizard")
            io.write("  1) Custom traffic on default qualified physics")
            io.write("  2) Custom physical configuration (supported exploratory domain)")
            io.write("  3) Load an existing YAML configuration")
            io.write("  4) Robustness campaign (A89/A90/A91 guided workflow)")
            choice = io.read("Choose 1/2/3/4 or Q to cancel > ").strip().lower()
            if choice in {"q", "quit", "cancel"}:
                raise WizardCancelled
            selected = {"1": "traffic", "2": "physics", "3": "load", "4": "robustness"}.get(choice)
            if selected is None:
                io.write("Unknown choice.")
                return 2

        if selected == "robustness":
            from .robustness_wizard import run_robustness_wizard

            return run_robustness_wizard(
                io=io,
                config_path=config_path,
                output_dir=output_dir,
                execute_unit=robustness_execute_unit,
                execution_code_identity_sha256=robustness_execution_code_identity_sha256,
                open_result=open_result or _default_open_result,
            )

        if selected == "load":
            raw = io.read("YAML path > ").strip()
            if not raw:
                raise WizardCancelled
            destination = Path(raw)
            if not destination.is_file():
                io.write(f"Configuration not found: {destination}")
                return 2
        else:
            mapping = _new_scenario(io, selected)
            destination = Path(config_path or _default_config_path(selected))
            write_scenario(mapping, destination)
            io.write(f"Saved configuration: {destination}")

        preflight = preflight_scenario(destination)
        _summary(io, destination, preflight)
        if execute is None or not _yes(io, "Start analysis now?", default=False):
            io.write("Configuration is saved and preflighted. No simulation was started.")
            return 0

        effective_output = Path(output_dir) if output_dir is not None else Path("cross_psp_outputs") / destination.stem
        io.write(f"Result folder: {effective_output.resolve()}")
        status = _execute_with_heartbeat(io, execute, destination, effective_output, preflight)

        if status == 0:
            report = effective_output / "REPORT.html"
            summary = effective_output / "ANALYSIS_SUMMARY.md"
            primary = report if report.is_file() else summary if summary.is_file() else None
            io.write("ANALYSIS SUCCESSFUL")
            if primary is not None:
                resolved = primary.resolve()
                io.write(f"  primary result: {resolved}")
                io.write(f"  result folder: {effective_output.resolve()}")
                opener = open_result or _default_open_result
                try:
                    opener(resolved)
                    if sys.platform == "win32" or open_result is not None:
                        io.write(f"  opened: {resolved.name}")
                except OSError as exc:
                    io.write(f"  result could not be opened automatically: {exc}")
            else:
                io.write(f"  no REPORT.html/ANALYSIS_SUMMARY.md found; inspect: {effective_output.resolve()}")
            return 0

        io.write(f"ANALYSIS FAILED with status {status}")
        io.write(f"  inspect result folder: {effective_output.resolve()}")
        return status
    except WizardBack:
        io.write("Back requested. No new simulation was started.")
        return 3
    except WizardCancelled:
        io.write("Cancelled. No new simulation was started.")
        return 0
    except Exception as exc:
        io.write(f"FAIL-CLOSED: {exc}")
        return 2
