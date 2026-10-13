from pathlib import Path

from cross_psp_two_stream.wizard import ConsoleIO, run_wizard
from cross_psp_two_stream.wizard_config import (
    StreamPhysicsInput,
    TrafficInput,
    build_physics_scenario,
    build_traffic_scenario,
    preflight_scenario,
    write_scenario,
)


def _physics(stream_id: str, heading: float, *, speed: float = 6.0):
    return StreamPhysicsInput(
        stream_id=stream_id,
        source_to_center_nm=1.7 if stream_id == "alpha" else 1.3,
        center_to_exit_nm=1.2 if stream_id == "alpha" else 1.9,
        heading_deg=heading,
        cruise_speed_knots=speed,
        maximum_speed_knots=12.0,
        length_m=110.0,
        beam_m=18.0,
        max_acceleration_knots_per_min=2.0,
        max_deceleration_knots_per_min=2.0,
        max_turn_rate_deg_per_min=25.0,
        safety_domain={
            "type": "four_quadrant_m",
            "ahead_m": 210.0,
            "astern_m": 120.0,
            "port_m": 160.0,
            "starboard_m": 180.0,
        },
    )


def test_traffic_builder_changes_only_supported_user_surface(tmp_path):
    mapping = build_traffic_scenario(TrafficInput(alpha_rate_per_hour=7.0, beta_rate_per_hour=5.0, replications=3))
    path = write_scenario(mapping, tmp_path / "traffic.yaml")
    pf = preflight_scenario(path, cache_dir=tmp_path / "cache")
    assert pf.validation.rate_per_hour_by_stream == {"alpha": 7.0, "beta": 5.0}
    assert pf.diagnostic_replication_runs == 9
    assert pf.scope_status in {"QUALIFIED_PROFILE", "SUPPORTED_EXPLORATORY"}


def test_physics_builder_preflights_as_supported_exploratory(tmp_path):
    mapping = build_physics_scenario(
        TrafficInput(),
        _physics("alpha", 15.0, speed=6.5),
        _physics("beta", 285.0, speed=5.5),
        alpha_b6_share=0.7,
    )
    path = write_scenario(mapping, tmp_path / "physics.yaml")
    pf = preflight_scenario(path, cache_dir=tmp_path / "cache")
    assert pf.scope_status == "SUPPORTED_EXPLORATORY"
    assert pf.dynamic_reference_runs_if_cache_miss == 6
    assert pf.validation.physical_changed_paths


def test_nonorthogonal_physics_fails_closed_in_preflight(tmp_path):
    mapping = build_physics_scenario(
        TrafficInput(),
        _physics("alpha", 0.0),
        _physics("beta", 260.0),
    )
    path = write_scenario(mapping, tmp_path / "bad.yaml")
    try:
        preflight_scenario(path, cache_dir=tmp_path / "cache")
    except Exception as exc:
        assert "orthogonal" in str(exc).lower()
    else:
        raise AssertionError("non-orthogonal configuration must fail closed")


def test_wizard_cancel_starts_no_execution():
    answers = iter(["cancel"])
    out = []
    called = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    status = run_wizard(mode="traffic", io=io, execute=lambda p, o: called.append((p, o)) or 0)
    assert status == 0
    assert called == []
    assert any("Cancelled" in line for line in out)


def test_wizard_traffic_saves_preflights_and_can_decline_execution(tmp_path, monkeypatch):
    answers = iter([
        "", "", "", "", "", "", "", "",  # traffic defaults
        "n",  # start?
    ])
    out = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    config = tmp_path / "wizard.yaml"
    status = run_wizard(mode="traffic", io=io, config_path=config, output_dir=tmp_path / "out")
    assert status == 0
    assert config.is_file()
    assert any("Preflight PASS" in line for line in out)
    assert any("No simulation was started" in line for line in out)


def test_wizard_load_existing_config(tmp_path):
    path = write_scenario(build_traffic_scenario(TrafficInput()), tmp_path / "existing.yaml")
    answers = iter([str(path), "n"])
    out = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    status = run_wizard(mode="load", io=io)
    assert status == 0
    assert any("Preflight PASS" in line for line in out)


def test_wizard_back_returns_launcher_code_without_execution():
    answers = iter(["back"])
    out = []
    called = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    status = run_wizard(mode="traffic", io=io, execute=lambda p, o: called.append((p, o)) or 0)
    assert status == 3
    assert called == []
    assert any("Back requested" in line for line in out)


def test_wizard_execution_reports_campaign_progress(tmp_path):
    answers = iter(["", "", "", "", "", "", "", "", "y"])
    out = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    status = run_wizard(
        mode="traffic",
        io=io,
        config_path=tmp_path / "run.yaml",
        output_dir=tmp_path / "out",
        execute=lambda scenario, output: 0,
    )
    assert status == 0
    assert any("Execution progress: 0/1" in line for line in out)
    assert any("Execution progress: 1/1" in line for line in out)


def test_default_physics_builder_preserves_qualified_baseline_identity(tmp_path):
    def default_physics(stream_id: str, heading: float):
        return StreamPhysicsInput(
            stream_id=stream_id,
            source_to_center_nm=1.5,
            center_to_exit_nm=1.5,
            heading_deg=heading,
            cruise_speed_knots=6.0,
            maximum_speed_knots=10.0,
            length_m=120.0,
            beam_m=20.0,
            max_acceleration_knots_per_min=2.0,
            max_deceleration_knots_per_min=2.0,
            max_turn_rate_deg_per_min=30.0,
            safety_domain={"type": "circular_nm", "radius_nm": 0.10},
        )

    traffic_path = write_scenario(build_traffic_scenario(TrafficInput()), tmp_path / "traffic.yaml")
    physics_path = write_scenario(
        build_physics_scenario(
            TrafficInput(),
            default_physics("alpha", 0.0),
            default_physics("beta", 270.0),
        ),
        tmp_path / "physics.yaml",
    )
    traffic_pf = preflight_scenario(traffic_path, cache_dir=tmp_path / "cache")
    physics_pf = preflight_scenario(physics_path, cache_dir=tmp_path / "cache")
    assert physics_pf.scope_status == "QUALIFIED_PROFILE"
    assert physics_pf.physics_identity_sha256 == traffic_pf.physics_identity_sha256
    assert physics_pf.validation.physical_changed_paths == ()


def test_identical_changed_stream_physics_uses_one_shared_class():
    alpha = _physics("alpha", 0.0)
    beta = StreamPhysicsInput(
        stream_id="beta",
        source_to_center_nm=alpha.source_to_center_nm,
        center_to_exit_nm=alpha.center_to_exit_nm,
        heading_deg=270.0,
        cruise_speed_knots=alpha.cruise_speed_knots,
        maximum_speed_knots=alpha.maximum_speed_knots,
        length_m=alpha.length_m,
        beam_m=alpha.beam_m,
        max_acceleration_knots_per_min=alpha.max_acceleration_knots_per_min,
        max_deceleration_knots_per_min=alpha.max_deceleration_knots_per_min,
        max_turn_rate_deg_per_min=alpha.max_turn_rate_deg_per_min,
        safety_domain=alpha.safety_domain,
    )
    mapping = build_physics_scenario(TrafficInput(), alpha, beta)
    classes = mapping["model"]["parameters"]["vessel_classes"]
    streams = mapping["model"]["parameters"]["streams"]
    assert list(classes) == ["wizard-shared"]
    assert {row["vessel_class_id"] for row in streams} == {"wizard-shared"}


def test_wizard_default_output_isolated_by_saved_config_and_opens_primary_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "", "", "", "", "", "", "y"])
    out = []
    opened = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=out.append)
    config = Path("cross_custom_configs") / "case-a.yaml"

    def execute(scenario, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / "REPORT.html").write_text("<html>ok</html>", encoding="utf-8")
        return 0

    status = run_wizard(
        mode="traffic",
        io=io,
        config_path=config,
        execute=execute,
        open_result=opened.append,
    )
    assert status == 0
    expected = (tmp_path / "cross_psp_outputs" / "case-a" / "REPORT.html").resolve()
    assert opened == [expected]
    assert any("ANALYSIS SUCCESSFUL" in line for line in out)
    assert any(str(expected) in line for line in out)


def test_loading_same_yaml_reuses_same_default_result_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = write_scenario(build_traffic_scenario(TrafficInput()), tmp_path / "saved-case.yaml")
    outputs = []

    def execute(scenario, output):
        outputs.append(Path(output).resolve())
        output.mkdir(parents=True, exist_ok=True)
        (output / "REPORT.html").write_text("<html>ok</html>", encoding="utf-8")
        return 0

    for _ in range(2):
        answers = iter([str(config), "y"])
        io = ConsoleIO(read=lambda prompt: next(answers), write=lambda text: None)
        assert run_wizard(mode="load", io=io, execute=execute, open_result=lambda p: None) == 0

    assert outputs == [
        (tmp_path / "cross_psp_outputs" / "saved-case").resolve(),
        (tmp_path / "cross_psp_outputs" / "saved-case").resolve(),
    ]


def test_default_windows_opener_opens_folder_then_primary(tmp_path, monkeypatch):
    import os
    import cross_psp_two_stream.wizard as wizard

    report = tmp_path / "run" / "REPORT.html"
    report.parent.mkdir()
    report.write_text("<html>ok</html>", encoding="utf-8")
    opened = []
    monkeypatch.setattr(wizard.sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", lambda value: opened.append(Path(value)), raising=False)

    wizard._default_open_result(report.resolve())

    assert opened == [report.parent.resolve(), report.resolve()]
