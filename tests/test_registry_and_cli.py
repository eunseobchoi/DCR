import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcr.errors import DCRError
from dcr.registry import authorize_external_evaluation, generate_runs, validate_runs, write_evaluation_lock

SRC = str(Path(__file__).resolve().parents[1] / "src")
LOCK = {"model_checkpoint_sha256": "a" * 64, "beta": 0.5, "threshold": 0.5, "postprocessing": "none",
        "split_sha256": "b" * 64, "config_sha256": "c" * 64, "code_commit": "deadbeef"}


def test_run_matrix_counts():
    runs = generate_runs()
    counts = validate_runs(runs, expect_total=90)
    assert dict(counts) == {"BASE": 36, "MAIN": 21, "REF": 9, "ABL": 24}
    assert "BASE-unet_r34-leave_C1_out-S00" in {r["run_id"] for r in runs}
    with pytest.raises(DCRError):
        validate_runs(runs + runs[:1])


def test_external_evaluation_is_one_shot(tmp_path):
    lock = tmp_path / "evaluation_lock.json"
    with pytest.raises(DCRError) as exc:
        authorize_external_evaluation(lock, tmp_path)
    assert exc.value.code == "EVALUATION_NOT_LOCKED"
    with pytest.raises(DCRError):
        write_evaluation_lock(tmp_path / "partial.json", {**LOCK, "beta": None})
    write_evaluation_lock(lock, LOCK)
    authorize_external_evaluation(lock, tmp_path)
    with pytest.raises(DCRError) as exc:
        authorize_external_evaluation(lock, tmp_path)
    assert exc.value.code == "EXTERNAL_ALREADY_EVALUATED"


def test_tampered_lock_rejected(tmp_path):
    lock = tmp_path / "lock.json"
    write_evaluation_lock(lock, LOCK)
    data = json.loads(lock.read_text())
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({**data, "beta": 1.0}))
    with pytest.raises(DCRError):
        authorize_external_evaluation(tampered, tmp_path)


def _cli(*args):
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run([sys.executable, "-m", "dcr.cli", *args], capture_output=True, text=True, env=env)


def test_cli_geometry_check():
    res = _cli("geometry", "check")
    assert res.returncode == 0, res.stderr
    assert res.stdout.count("PASS") == 4


def test_cli_synthetic_train_and_resume(tmp_path):
    out = tmp_path / "run"
    res = _cli("train", "refiner", "--method", "M07", "--synthetic", "--steps", "3",
               "--batch-size", "2", "--size", "64", "--out", str(out))
    assert res.returncode == 0, res.stderr
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "COMPLETED" and record["optimizer_updates_completed"] == 3
    res = _cli("train", "refiner", "--method", "M07", "--synthetic", "--steps", "5",
               "--batch-size", "2", "--size", "64", "--out", str(out))
    assert res.returncode == 0, res.stderr
    assert json.loads((out / "run.json").read_text())["optimizer_updates_completed"] == 5


def test_cli_external_without_lock_fails(tmp_path):
    res = _cli("evaluate", "external", "--lock", str(tmp_path / "missing.json"), "--marker-dir", str(tmp_path))
    assert res.returncode == 2 and "EVALUATION_NOT_LOCKED" in res.stderr
