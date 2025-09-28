🧠 ReMemory — Reconstructive Episodic Memory for LLMs

🇷🇺 Read this in Russian: ./README_RU.md

────────────────────────────

📌 Concept

ReMemory is an experimental prototype of a human-like memory architecture for large language models (LLMs).
Instead of storing and retrieving raw text data, ReMemory encodes episodes as semantic functions that can be reconstructed from meaning.
This approach moves beyond database-like retrieval — it allows an agent to remember experiences contextually and associatively, much like a human does.

────────────────────────────

📦 Installation

Requirements:

Python 3.10+

PyTorch

sentence-transformers

numpy

Install dependencies:

pip install -r requirements.txt

────────────────────────────

📁 Project Structure

ReMemory/
├─ train_memory.py # Train memory cells on a JSON dataset
├─ scripts/
│ └─ recall_memory.py # Retrieve episodes from semantic keys
└─ examples/
└─ system_prompt.py # System prompt example for LLM integration

────────────────────────────

🧠 Memory Training

To train the memory, create a JSON dataset containing episodes with keywords (semantic keys) and text (content). Example:

[
{
"keywords": "Ilya, bridge, night",
"text": "We walked with Ilya across the bridge when the city lights turned on..."
},
{
"keywords": "Sergey, Gazelle, repair",
"text": "Sergey brought an old Gazelle van with a stuck window..."
}
]

Then run:

python train_memory.py

✅ Once training is complete, all episodes will be stored as memory weights.
You can safely delete the original JSON file — it is no longer required for retrieval.

────────────────────────────

🔎 Memory Recall

After training, you can query the memory with a semantic phrase:

python scripts/recall_memory.py --top_k 3

Example interaction:

🔎 Enter a phrase: walking on a bridge with Ilya

Result:

📁 ID: cell_0003
📈 Similarity: 0.942

🧠 Reconstructed memory:
We walked with Ilya across the bridge when the city lights turned on...

📊 Closest matches:

cell_0003 — score=0.942

cell_0005 — score=0.812

cell_0002 — score=0.785

────────────────────────────

🤖 Integration with LLM

ReMemory can be integrated with an LLM by injecting recalled episodes into the system prompt before each dialogue turn.

Example system_prompt.py:

You are an intelligent agent equipped with reconstructive episodic memory.
Your task is not only to respond but to remember past experiences.
You receive a list of memory episodes and may use them to enrich the context of the dialogue.
If the memory is incomplete or fragmentary, that's normal — humans remember in fragments too.

📌 Combine this prompt with retrieved episodes to give your LLM a sense of continuity and past experience.

────────────────────────────

📚 Paper

The theoretical foundations and architectural design of ReMemory are described in detail in the accompanying paper:
Reconstructive Episodic Memory (zenodo preprint):[https://https://zenodo.org/records/17220514]
────────────────────────────

✍️ Author

Mikhail Smirnov, ReMemory Project (26.09.2025)

ReMemory is a research prototype demonstrating that memory in AI can be more than a database — it can become an active cognitive process.

────────────────────────────

📜 License

This project and its associated materials are registered with Safe Creative to protect authorship and intellectual property.

Author: Mikhail Smirnov
Registration date: 26.09.2025
