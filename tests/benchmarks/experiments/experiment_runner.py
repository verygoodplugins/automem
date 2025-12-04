#!/usr/bin/env python3
"""
AutoMem Experiment Runner

Automated experimentation framework for finding optimal memory configurations.

Usage:
    python experiment_runner.py --mode grid      # Run predefined grid
    python experiment_runner.py --mode explore   # AI-guided exploration
    python experiment_runner.py --mode ablation  # Ablation study on one param
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiment_config import (
    ExperimentConfig, 
    ExperimentResult,
    generate_experiment_grid,
    generate_focused_experiments,
)
from railway_manager import RailwayManager, LocalTestManager, RailwayInstance

# Import the benchmark runner
from tests.benchmarks.test_locomo import LoCoMoEvaluator, LoCoMoConfig


class ExperimentRunner:
    """
    Orchestrates experiments across multiple configurations.
    """
    
    def __init__(
        self,
        results_dir: str = "experiment_results",
        use_railway: bool = False,
        max_parallel: int = 2,
        questions_per_config: Optional[int] = None,  # None = full benchmark
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_railway = use_railway
        self.max_parallel = max_parallel
        self.questions_per_config = questions_per_config
        
        if use_railway:
            self.instance_manager = RailwayManager(max_concurrent_instances=max_parallel)
        else:
            self.local_manager = LocalTestManager()
        
        self.results: List[ExperimentResult] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def run_single_experiment(
        self,
        config: ExperimentConfig,
        instance: Optional[RailwayInstance] = None,
    ) -> ExperimentResult:
        """Run benchmark against a single configuration"""
        print(f"\n{'='*60}")
        print(f"🧪 Running experiment: {config.name}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Determine the API URL
        if instance:
            base_url = instance.url
        else:
            base_url = os.getenv("AUTOMEM_TEST_BASE_URL", "http://localhost:8001")
        
        # Configure the benchmark
        locomo_config = LoCoMoConfig(
            base_url=base_url,
            api_token=os.getenv("AUTOMEM_TEST_API_TOKEN", "test-token"),
            recall_limit=config.recall_limit,
            eval_mode="e2e",
            use_official_f1=True,
            use_lenient_eval=True,
            e2e_model="gpt-5.1",
            eval_judge_model="gpt-5.1",
        )
        
        # Run the benchmark
        evaluator = LoCoMoEvaluator(locomo_config)
        
        try:
            # If we're doing quick tests, limit to first conversation
            if self.questions_per_config:
                # Run on subset (first conversation only)
                # Get first conversation ID from dataset
                import json
                with open(locomo_config.data_file, 'r') as f:
                    convs = json.load(f)
                first_conv_id = convs[0].get('sample_id') if convs else None
                results = evaluator.run_benchmark(
                    cleanup_after=True,
                    conversation_ids=[first_conv_id] if first_conv_id else None,
                )
            else:
                results = evaluator.run_benchmark(cleanup_after=True)
            
            # Extract metrics
            experiment_result = ExperimentResult(
                config=config,
                overall_accuracy=results["overall"]["accuracy"],
                single_hop_accuracy=results.get("category_1", {}).get("accuracy", 0),
                temporal_accuracy=results.get("category_2", {}).get("accuracy", 0),
                multi_hop_accuracy=results.get("category_3", {}).get("accuracy", 0),
                open_domain_accuracy=results.get("category_4", {}).get("accuracy", 0),
                adversarial_accuracy=results.get("category_5", {}).get("accuracy", 0),
                overall_f1=results.get("official_metrics", {}).get("overall_mean_f1", 0),
                avg_response_time_ms=results.get("performance", {}).get("avg_recall_ms", 0),
                run_timestamp=start_time.isoformat(),
                run_duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            experiment_result = ExperimentResult(
                config=config,
                errors=[str(e)],
                run_timestamp=start_time.isoformat(),
                run_duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
        
        # Save individual result
        self._save_result(experiment_result)
        self.results.append(experiment_result)
        
        return experiment_result
    
    async def run_grid_search(
        self,
        configs: Optional[List[ExperimentConfig]] = None,
    ) -> List[ExperimentResult]:
        """Run all configurations in the grid"""
        if configs is None:
            configs = generate_experiment_grid()
        
        print(f"\n🔬 Starting grid search with {len(configs)} configurations")
        
        if self.use_railway:
            # Deploy instances and run in parallel
            tasks = []
            for config in configs:
                tasks.append(self._run_with_railway(config))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            return [r for r in results if isinstance(r, ExperimentResult)]
        else:
            # Run sequentially against local instance
            results = []
            for config in configs:
                # Apply config if possible
                await self.local_manager.apply_config(config)
                result = await self.run_single_experiment(config)
                results.append(result)
            return results
    
    async def run_exploration(
        self,
        max_iterations: int = 20,
        convergence_threshold: float = 0.01,
    ) -> ExperimentConfig:
        """
        AI-guided exploration to find optimal configuration.
        Uses a simple evolutionary strategy.
        """
        print(f"\n🤖 Starting AI-guided exploration (max {max_iterations} iterations)")
        
        # Start with baseline
        current_best = generate_experiment_grid()[0]
        best_score = 0.0
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Generate mutations of current best
            candidates = self._generate_mutations(current_best, num_mutations=3)
            candidates.append(current_best)  # Always include current best
            
            # Evaluate candidates
            results = await self.run_grid_search(candidates)
            
            # Find best result
            if results:
                scored_results = [(r, r.score()) for r in results]
                scored_results.sort(key=lambda x: x[1], reverse=True)
                best_result, score = scored_results[0]
                
                improvement = score - best_score
                print(f"📊 Best score this iteration: {score:.4f} (improvement: {improvement:+.4f})")
                
                if improvement > convergence_threshold:
                    current_best = best_result.config
                    best_score = score
                    no_improvement_count = 0
                    print(f"✨ New best configuration: {current_best.name}")
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= 3:
                    print(f"🎯 Converged after {iteration + 1} iterations")
                    break
        
        print(f"\n🏆 Best configuration: {current_best.name}")
        print(f"   Score: {best_score:.4f}")
        return current_best
    
    async def run_ablation_study(
        self,
        param_name: str,
        values: List[Any],
        base_config: Optional[ExperimentConfig] = None,
    ) -> List[ExperimentResult]:
        """
        Ablation study: vary one parameter while holding others constant.
        """
        if base_config is None:
            base_config = generate_experiment_grid()[0]
        
        print(f"\n📉 Running ablation study on: {param_name}")
        print(f"   Values to test: {values}")
        
        configs = generate_focused_experiments(base_config, param_name, values)
        return await self.run_grid_search(configs)
    
    async def _run_with_railway(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with a Railway deployment"""
        instance = await self.instance_manager.deploy_instance(config)
        
        if instance is None or instance.status != "healthy":
            return ExperimentResult(
                config=config,
                errors=["Failed to deploy instance"],
                run_timestamp=datetime.now().isoformat(),
            )
        
        try:
            result = await self.run_single_experiment(config, instance)
            return result
        finally:
            await self.instance_manager.destroy_instance(instance)
    
    def _generate_mutations(
        self,
        config: ExperimentConfig,
        num_mutations: int = 3,
    ) -> List[ExperimentConfig]:
        """Generate mutated configurations for exploration"""
        mutations = []
        
        # Define mutation space
        mutation_options = {
            "recall_limit": [5, 10, 15, 20, 30],
            "vector_weight": [0.3, 0.4, 0.5, 0.6, 0.7],
            "graph_weight": [0.3, 0.4, 0.5, 0.6, 0.7],
            "graph_expansion_depth": [1, 2, 3, 4],
            "enrichment_model": ["gpt-4o-mini", "gpt-4.1", "gpt-5.1"],
            "embedding_model": ["text-embedding-3-small", "text-embedding-3-large"],
        }
        
        for i in range(num_mutations):
            # Pick random parameter to mutate
            param = random.choice(list(mutation_options.keys()))
            value = random.choice(mutation_options[param])
            
            # Create mutated config
            config_dict = config.__dict__.copy()
            config_dict["name"] = f"{config.name}_mut{i}_{param}"
            config_dict[param] = value
            
            # Keep weights normalized
            if param in ["vector_weight", "graph_weight"]:
                other = "graph_weight" if param == "vector_weight" else "vector_weight"
                config_dict[other] = 1.0 - value
            
            mutations.append(ExperimentConfig(**config_dict))
        
        return mutations
    
    def _save_result(self, result: ExperimentResult):
        """Save a single result to disk"""
        filename = f"{self.run_id}_{result.config.name}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump({
                "config": result.config.__dict__,
                "metrics": {
                    "overall_accuracy": result.overall_accuracy,
                    "single_hop": result.single_hop_accuracy,
                    "temporal": result.temporal_accuracy,
                    "multi_hop": result.multi_hop_accuracy,
                    "open_domain": result.open_domain_accuracy,
                    "adversarial": result.adversarial_accuracy,
                    "f1": result.overall_f1,
                    "response_time_ms": result.avg_response_time_ms,
                    "score": result.score(),
                },
                "metadata": {
                    "timestamp": result.run_timestamp,
                    "duration_seconds": result.run_duration_seconds,
                    "errors": result.errors,
                }
            }, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a summary report of all experiments"""
        if not self.results:
            return "No results to report."
        
        # Sort by score
        sorted_results = sorted(self.results, key=lambda r: r.score(), reverse=True)
        
        report = []
        report.append("=" * 70)
        report.append("📊 EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 70)
        report.append(f"Run ID: {self.run_id}")
        report.append(f"Total experiments: {len(self.results)}")
        report.append("")
        
        # Top 5 configurations
        report.append("🏆 TOP CONFIGURATIONS:")
        report.append("-" * 70)
        for i, result in enumerate(sorted_results[:5]):
            report.append(f"\n{i+1}. {result.config.name}")
            report.append(f"   Score: {result.score():.4f}")
            report.append(f"   Accuracy: {result.overall_accuracy:.2%}")
            report.append(f"   Multi-hop: {result.multi_hop_accuracy:.2%}")
            report.append(f"   Temporal: {result.temporal_accuracy:.2%}")
            report.append(f"   Response Time: {result.avg_response_time_ms:.0f}ms")
        
        # Category breakdown for best config
        best = sorted_results[0]
        report.append("\n" + "=" * 70)
        report.append(f"📈 BEST CONFIG DETAILS: {best.config.name}")
        report.append("=" * 70)
        report.append(f"Embedding: {best.config.embedding_model}")
        report.append(f"Enrichment: {best.config.enrichment_model}")
        report.append(f"Recall Strategy: {best.config.recall_strategy}")
        report.append(f"Recall Limit: {best.config.recall_limit}")
        report.append(f"Vector Weight: {best.config.vector_weight}")
        report.append(f"Graph Weight: {best.config.graph_weight}")
        report.append(f"Extract Facts: {best.config.extract_facts}")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.results_dir / f"{self.run_id}_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        return report_text


async def main():
    parser = argparse.ArgumentParser(
        description="AutoMem Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick local test with subset of questions
  python experiment_runner.py --mode grid --quick

  # Full grid search
  python experiment_runner.py --mode grid

  # AI-guided exploration
  python experiment_runner.py --mode explore --iterations 10

  # Ablation study on recall_limit
  python experiment_runner.py --mode ablation --param recall_limit --values 5,10,15,20

  # Use Railway for parallel deployment
  python experiment_runner.py --mode grid --railway --parallel 3
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["grid", "explore", "ablation"],
        default="grid",
        help="Experiment mode"
    )
    parser.add_argument(
        "--railway",
        action="store_true",
        help="Use Railway for deployment (otherwise local)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Max parallel instances"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with limited questions"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Max iterations for exploration mode"
    )
    parser.add_argument(
        "--param",
        type=str,
        help="Parameter for ablation study"
    )
    parser.add_argument(
        "--values",
        type=str,
        help="Comma-separated values for ablation study"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory for results"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(
        results_dir=args.output_dir,
        use_railway=args.railway,
        max_parallel=args.parallel,
        questions_per_config=100 if args.quick else None,
    )
    
    try:
        if args.mode == "grid":
            results = await runner.run_grid_search()
        
        elif args.mode == "explore":
            best_config = await runner.run_exploration(
                max_iterations=args.iterations
            )
            print(f"\n🎯 Optimal configuration found:")
            print(best_config.to_json())
        
        elif args.mode == "ablation":
            if not args.param or not args.values:
                print("Error: --param and --values required for ablation mode")
                sys.exit(1)
            
            values = [v.strip() for v in args.values.split(",")]
            # Try to convert to appropriate types
            try:
                values = [int(v) for v in values]
            except ValueError:
                try:
                    values = [float(v) for v in values]
                except ValueError:
                    pass  # Keep as strings
            
            results = await runner.run_ablation_study(args.param, values)
        
        # Generate and print report
        report = runner.generate_report()
        print(report)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Cleaning up...")
        if args.railway:
            await runner.instance_manager.destroy_all()
    
    finally:
        if args.railway:
            await runner.instance_manager.destroy_all()


if __name__ == "__main__":
    asyncio.run(main())

