"""
LongMemEval Benchmark Configuration Variants

Defines different recall and storage configurations to test AutoMem's
performance across different settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LongMemEvalConfig:
    """Configuration for LongMemEval benchmark evaluation."""

    # AutoMem API settings
    base_url: str = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
    api_token: str = os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token")

    # Dataset paths
    data_file: str = str(Path(__file__).parent / "data" / "longmemeval_s_cleaned.json")
    oracle_file: str = str(Path(__file__).parent / "data" / "longmemeval_oracle.json")
    results_dir: str = str(Path(__file__).parent / "results")

    # Storage strategy: "per-session" or "per-turn"
    storage_strategy: str = "per-session"

    # Recall settings
    recall_limit: int = 10
    expand_entities: bool = False
    expand_relations: bool = False
    auto_decompose: bool = False
    use_temporal_hints: bool = False

    # Memory importance
    importance: float = 0.5

    # Answer generation
    llm_model: str = os.getenv("LONGMEMEVAL_LLM_MODEL", "gpt-4o")
    eval_llm_model: Optional[str] = os.getenv("LONGMEMEVAL_EVAL_LLM_MODEL")
    use_chain_of_note: bool = True

    # Performance tuning
    batch_size: int = 20  # Sessions to store before pausing
    pause_between_batches: float = 0.3
    request_timeout: int = 30

    # Evaluation
    use_llm_eval: bool = False  # Use GPT-4o for evaluation (costs money)
    max_questions: int = 0  # 0 = all questions

    # Tag prefix for cleanup
    tag_prefix: str = "longmemeval"

    # Config name for results tracking
    name: str = "baseline"


# Pre-defined benchmark configurations
BENCHMARK_CONFIGS: Dict[str, dict] = {
    "baseline": {
        "name": "baseline",
        "storage_strategy": "per-session",
        "recall_limit": 10,
        "expand_entities": False,
        "expand_relations": False,
        "auto_decompose": False,
        "use_temporal_hints": False,
    },
    "per-turn": {
        "name": "per-turn",
        "storage_strategy": "per-turn",
        "recall_limit": 10,
        "expand_entities": False,
        "expand_relations": False,
        "auto_decompose": False,
        "use_temporal_hints": False,
    },
    "expand-entities": {
        "name": "expand-entities",
        "storage_strategy": "per-session",
        "recall_limit": 10,
        "expand_entities": True,
        "expand_relations": False,
        "auto_decompose": False,
        "use_temporal_hints": False,
    },
    "expand-relations": {
        "name": "expand-relations",
        "storage_strategy": "per-session",
        "recall_limit": 10,
        "expand_entities": False,
        "expand_relations": True,
        "auto_decompose": False,
        "use_temporal_hints": False,
    },
    "high-k": {
        "name": "high-k",
        "storage_strategy": "per-session",
        "recall_limit": 20,
        "expand_entities": False,
        "expand_relations": False,
        "auto_decompose": False,
        "use_temporal_hints": False,
    },
    "temporal": {
        "name": "temporal",
        "storage_strategy": "per-session",
        "recall_limit": 10,
        "expand_entities": False,
        "expand_relations": False,
        "auto_decompose": False,
        "use_temporal_hints": True,
    },
    "full-graph": {
        "name": "full-graph",
        "storage_strategy": "per-session",
        "recall_limit": 20,
        "expand_entities": True,
        "expand_relations": True,
        "auto_decompose": True,
        "use_temporal_hints": True,
    },
}


def get_config(name: str = "baseline", **overrides) -> LongMemEvalConfig:
    """Get a benchmark config by name with optional overrides."""
    if name in BENCHMARK_CONFIGS:
        params = {**BENCHMARK_CONFIGS[name], **overrides}
    else:
        params = {"name": name, **overrides}
    return LongMemEvalConfig(**params)
