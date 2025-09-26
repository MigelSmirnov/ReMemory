# memory/mlp_core/mlp_trainer.py
"""
Train a small MLP to reconstruct token IDs from a semantic vector.
The training continues until loss <= 1e-5 (precision target) or max_epochs is reached.
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from memory.mlp_core.mininet_regression import MiniNetRegression
from memory.common_paths import CELLS_DIR


def train_cell(
    context_vector: list[float],
    token_ids: list[int],
    cell_id: str,
    save_dir: Path = CELLS_DIR,
    epochs: int = 2000,
    lr: float = 0.01,
    target_loss: float = 1e-5
) -> dict:
    """
    Train a small MLP to learn mapping from context_vector -> token_ids.

    Args:
        context_vector: Semantic embedding vector (input).
        token_ids: Encoded token sequence (target).
        cell_id: Unique ID for this memory cell (e.g. "vec_0001").
        save_dir: Directory where model and config will be saved.
        epochs: Maximum training epochs (safety limit).
        lr: Learning rate.
        target_loss: Stop training when loss <= this threshold.

    Returns:
        dict: model metadata (path, epochs, final_loss).
    """

    # 📏 Define network dimensions dynamically
    input_dim = len(context_vector)
    output_dim = len(token_ids)
    hidden_dim = max(128, (input_dim + output_dim) // 2)

    # 🧠 Initialize model
    model = MiniNetRegression(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor([context_vector], dtype=torch.float32)
    y = torch.tensor([token_ids], dtype=torch.float32)

    actual_epochs = 0
    final_loss_value = None
    reached_target = False

    # 🔁 Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        actual_epochs = epoch + 1
        final_loss_value = float(loss.item())

        if final_loss_value <= target_loss:
            reached_target = True
            print(f"✅ Target loss reached ({final_loss_value:.8f}) at epoch {actual_epochs}")
            break

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[Epoch {actual_epochs}/{epochs}] Loss: {final_loss_value:.8f}")

    if not reached_target:
        print(f"⚠️ Target loss {target_loss} not reached after {epochs} epochs (final: {final_loss_value:.8f})")

    # 💾 Save model
    cell_path = Path(save_dir) / cell_id
    cell_path.mkdir(parents=True, exist_ok=True)
    model_path = cell_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    # 📄 Save metadata and architecture
    model_config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "epochs": epochs,
        "target_loss": target_loss,
        "actual_epochs": actual_epochs,
        "final_loss": final_loss_value
    }

    with open(cell_path / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    return {
        "model_path": str(model_path),
        "actual_epochs": actual_epochs,
        "final_loss": final_loss_value,
        "reached_target": reached_target
    }
