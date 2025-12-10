"""
AutoMem Experiment Configuration System

Defines configuration variants to test for optimal memory retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import itertools
import json


class EmbeddingModel(Enum):
    """Embedding model options"""
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"


class EnrichmentModel(Enum):
    """LLM model for enrichment/extraction"""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    GPT_5_1 = "gpt-5.1"
    CLAUDE_HAIKU = "claude-3-haiku"


class RecallStrategy(Enum):
    """Memory recall strategy"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass
class ExperimentConfig:
    """A single experiment configuration"""
    name: str
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    
    # Enrichment settings
    enrichment_model: str = "gpt-4o-mini"
    enrichment_enabled: bool = True
    extract_entities: bool = True
    extract_topics: bool = True
    extract_facts: bool = False  # NEW: Statement-level extraction
    
    # Recall settings
    recall_strategy: str = "hybrid"
    recall_limit: int = 10
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    recency_weight: float = 0.1
    
    # Graph settings
    graph_expansion_depth: int = 2
    use_entity_linking: bool = True
    
    # Performance settings
    batch_size: int = 50
    semaphore_limit: int = 10
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert config to Railway environment variables"""
        return {
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_DIM": str(self.embedding_dim),
            "ENRICHMENT_MODEL": self.enrichment_model,
            "ENRICHMENT_ENABLED": str(self.enrichment_enabled).lower(),
            "EXTRACT_ENTITIES": str(self.extract_entities).lower(),
            "EXTRACT_TOPICS": str(self.extract_topics).lower(),
            "EXTRACT_FACTS": str(self.extract_facts).lower(),
            "RECALL_STRATEGY": self.recall_strategy,
            "RECALL_LIMIT": str(self.recall_limit),
            "VECTOR_WEIGHT": str(self.vector_weight),
            "GRAPH_WEIGHT": str(self.graph_weight),
            "RECENCY_WEIGHT": str(self.recency_weight),
            "GRAPH_EXPANSION_DEPTH": str(self.graph_expansion_depth),
            "USE_ENTITY_LINKING": str(self.use_entity_linking).lower(),
            "BATCH_SIZE": str(self.batch_size),
            "SEMAPHORE_LIMIT": str(self.semaphore_limit),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    config: ExperimentConfig
    
    # Accuracy metrics
    overall_accuracy: float = 0.0
    single_hop_accuracy: float = 0.0
    temporal_accuracy: float = 0.0
    multi_hop_accuracy: float = 0.0
    open_domain_accuracy: float = 0.0
    adversarial_accuracy: float = 0.0
    
    # F1 metrics
    overall_f1: float = 0.0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    
    # Quality metrics (subjective)
    retrieval_relevance: float = 0.0  # How relevant were retrieved memories?
    answer_coherence: float = 0.0     # How coherent were generated answers?
    
    # Metadata
    run_timestamp: str = ""
    run_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted score for this configuration.
        Higher is better.
        """
        if weights is None:
            weights = {
                "accuracy": 0.4,
                "response_time": 0.2,
                "multi_hop": 0.2,  # Emphasize our weakness
                "cost": 0.1,
                "temporal": 0.1,
            }
        
        # Normalize response time (lower is better, cap at 1000ms)
        response_score = max(0, 1 - (self.avg_response_time_ms / 1000))
        
        # Normalize cost (lower is better, cap at $10)
        cost_score = max(0, 1 - (self.total_cost_usd / 10))
        
        return (
            weights["accuracy"] * self.overall_accuracy +
            weights["response_time"] * response_score +
            weights["multi_hop"] * self.multi_hop_accuracy +
            weights["cost"] * cost_score +
            weights["temporal"] * self.temporal_accuracy
        )


def generate_experiment_grid() -> List[ExperimentConfig]:
    """
    Generate a grid of experiments to try.
    Uses a smart subset rather than full combinatorial explosion.
    """
    configs = []
    
    # Baseline config
    configs.append(ExperimentConfig(
        name="baseline",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4o-mini",
        recall_strategy="hybrid",
        recall_limit=10,
        vector_weight=0.6,
        graph_weight=0.4,
    ))
    
    # Test larger embedding model
    configs.append(ExperimentConfig(
        name="large_embeddings",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072,
        enrichment_model="gpt-4o-mini",
        recall_strategy="hybrid",
    ))
    
    # Test better enrichment model
    configs.append(ExperimentConfig(
        name="better_enrichment",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4.1",
        recall_strategy="hybrid",
    ))
    
    # Test fact extraction (new feature)
    configs.append(ExperimentConfig(
        name="with_facts",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4.1",
        extract_facts=True,
        recall_strategy="hybrid",
    ))
    
    # Test more aggressive recall
    configs.append(ExperimentConfig(
        name="more_recall",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4o-mini",
        recall_limit=20,
        recall_strategy="hybrid",
    ))
    
    # Test graph-heavy strategy
    configs.append(ExperimentConfig(
        name="graph_heavy",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4o-mini",
        recall_strategy="hybrid_weighted",
        vector_weight=0.4,
        graph_weight=0.6,
        graph_expansion_depth=3,
    ))
    
    # Test vector-only (for comparison)
    configs.append(ExperimentConfig(
        name="vector_only",
        embedding_model="text-embedding-3-small",
        enrichment_model="gpt-4o-mini",
        recall_strategy="vector_only",
        use_entity_linking=False,
    ))
    
    # Premium config (best everything, cost no object)
    configs.append(ExperimentConfig(
        name="premium",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072,
        enrichment_model="gpt-5.1",
        extract_facts=True,
        recall_limit=20,
        recall_strategy="hybrid_weighted",
        vector_weight=0.5,
        graph_weight=0.5,
        graph_expansion_depth=3,
    ))
    
    return configs


def generate_focused_experiments(
    base_config: ExperimentConfig,
    param_to_vary: str,
    values: List[Any]
) -> List[ExperimentConfig]:
    """
    Generate experiments varying a single parameter.
    Useful for ablation studies.
    """
    configs = []
    for i, value in enumerate(values):
        config_dict = base_config.__dict__.copy()
        config_dict["name"] = f"{base_config.name}_{param_to_vary}_{i}"
        config_dict[param_to_vary] = value
        configs.append(ExperimentConfig(**config_dict))
    return configs


