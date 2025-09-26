# memory/mlp_core/mlp_decoder.py

import torch
import json
import os
from memory.mlp_core.mininet_regression import MiniNetRegression

def reconstruct_token_ids(
    context_vector: list[float],
    model_path: str,
    config_path: str,
    token_range: tuple[int, int] = (0, 4095)
) -> list[int]:
    # 📁 читаем полный конфиг
    with open(config_path, "r", encoding="utf-8") as f:
        config_all = json.load(f)

    # ✅ оставляем только нужные ключи
    model_config = {
        k: config_all[k]
        for k in ("input_dim", "hidden_dim", "output_dim")
        if k in config_all
    }

    # 🧠 создаём модель только с теми параметрами, которые она понимает
    model = MiniNetRegression(**model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 🔁 прогоняем контекст через модель
    x = torch.tensor([context_vector], dtype=torch.float32)
    with torch.no_grad():
        output = model(x).squeeze(0).tolist()

    # 🧪 постобработка: ограничиваем токены диапазоном
    min_token, max_token = token_range
    token_ids = [max(min_token, min(max_token, int(round(v)))) for v in output]

    return token_ids


def reconstruct_from_saved_vector(cell_path: str, token_range: tuple[int, int] = (0, 4095)) -> list[int]:
    config_path = os.path.join(cell_path, "model_config.json")
    vector_path = os.path.join(cell_path, "context_vector.json")
    model_path = os.path.join(cell_path, "model.pt")

    with open(vector_path, "r", encoding="utf-8") as f:
        context_vector = json.load(f)

    return reconstruct_token_ids(context_vector, model_path, config_path, token_range)
