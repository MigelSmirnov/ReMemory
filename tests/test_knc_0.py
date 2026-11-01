import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "knc_0" / "run.py"
SPEC = importlib.util.spec_from_file_location("knc_0_run", MODULE_PATH)
knc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(knc)


def test_token_metrics_exact():
    truth = torch.tensor([[0, 42, 4095]])
    assert knc.metrics(truth.clone(), truth) == {
        "exact_fraction": 1.0,
        "all_exact": True,
        "mae_tokens": 0.0,
    }


def test_cuda_request_fails_cleanly_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        knc.resolve_device("cuda")


def test_smoke_run_has_clean_public_artifacts(tmp_path):
    args = knc.build_parser().parse_args([
        "--epochs", "200", "--probes", "16", "--input-dim", "8",
        "--hidden-dim", "8", "--output-dim", "4", "--check-every", "20",
        "--out", str(tmp_path), "--quiet",
    ])
    result = knc.run_trial(args, "smoke", 7, torch.device("cpu"))
    public_dir = tmp_path / "smoke" / "seed_7" / "public"
    assert {path.name for path in public_dir.iterdir()} == {"model.pt", "config.json"}
    assert "seed" not in (public_dir / "config.json").read_text()
    assert result["model_bytes"] > 0
