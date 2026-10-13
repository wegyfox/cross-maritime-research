from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT.parent
RUNNER = BUNDLE_ROOT / "acceptance" / "A97_PUBLIC_MATERIAL_CAMPAIGN_RUNNER.py"
LAUNCHER = BUNDLE_ROOT / "acceptance" / "RUN_A97_PUBLIC_MATERIAL_CAMPAIGN.ps1"
LIST_LAUNCHER = BUNDLE_ROOT / "acceptance" / "LIST_A97_PUBLIC_MATERIAL_CAMPAIGNS.ps1"
OUTPUT_ROOT = BUNDLE_ROOT / "acceptance" / "a97_material_campaign_work"
EXPECTED_IDS = (
    "fixed-safety-domain-interaction",
    "fixed-cruise-interaction",
    "fixed-maneuverability-interaction",
    "distribution-length-variability",
    "fleet-mixture-composition",
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("a97_public_campaign_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_list_is_exact_and_side_effect_free(tmp_path):
    before = set(OUTPUT_ROOT.glob("*")) if OUTPUT_ROOT.exists() else set()
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--list"],
        cwd=BUNDLE_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if ": priority=" in line]
    assert tuple(line.split(":", 1)[0] for line in lines) == EXPECTED_IDS
    after = set(OUTPUT_ROOT.glob("*")) if OUTPUT_ROOT.exists() else set()
    assert after == before


def test_campaign_resolution_is_exact_and_unknown_fails_closed():
    module = _load_runner_module()
    for campaign_id in EXPECTED_IDS:
        item, config = module._resolve_campaign(campaign_id)
        assert item["id"] == campaign_id
        assert config.is_file()
        assert config.parent == PROJECT_ROOT / "examples" / "robustness"
    with pytest.raises(SystemExit, match="unknown campaign id"):
        module._resolve_campaign("run-all")


def test_windows_launcher_requires_one_validated_campaign_and_has_no_run_all_default():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "[Parameter(Mandatory=$true, Position=0)]" in text
    assert "[ValidateSet(" in text
    for campaign_id in EXPECTED_IDS:
        assert f'"{campaign_id}"' in text
    assert "$CampaignId" in text
    assert '"--yes"' in text
    assert "RunAll" not in text
    assert "foreach" not in text.lower()


def test_runner_uses_shared_identity_safe_reference_cache_and_list_helper_is_nonexecuting():
    runner_text = RUNNER.read_text(encoding="utf-8")
    list_text = LIST_LAUNCHER.read_text(encoding="utf-8")
    assert 'REFERENCE_CACHE = BUNDLE_ROOT / "acceptance" / "a97_material_reference_cache"' in runner_text
    assert "reference_cache_dir=REFERENCE_CACHE" in runner_text
    assert "automatic_qualification_promotion=false" in runner_text
    assert "--list" in list_text
    assert "RUN_A97_PUBLIC_MATERIAL_CAMPAIGN.ps1" in list_text
