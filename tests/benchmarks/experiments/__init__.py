"""
AutoMem Experiment Framework

Automated experimentation for finding optimal memory configurations.
"""

from .experiment_config import (
    ExperimentConfig,
    ExperimentResult,
    generate_experiment_grid,
    generate_focused_experiments,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "generate_experiment_grid",
    "generate_focused_experiments",
]

