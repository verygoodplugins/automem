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

try:
    from .experiment_config import (
        ExperimentConfig, 
        ExperimentResult,
        generate_experiment_grid,
        generate_focused_experiments,
    )
    from .railway_manager import RailwayManager, LocalTestManager, RailwayInstance
except ImportError:
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
        num_conversations: int = 10,  # How many conversations to test (1=quick, 3=medium, 10=full)
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_railway = use_railway
        self.max_parallel = max_parallel
        self.questions_per_config = questions_per_config
        self.num_conversations = num_conversations
        
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
            # Select conversations based on num_conversations setting
            import json
            with open(locomo_config.data_file, 'r') as f:
                convs = json.load(f)
            
            # Get conversation IDs (limit to num_conversations)
            conv_ids = [c.get('sample_id') for c in convs[:self.num_conversations]]
            
            results = evaluator.run_benchmark(
                cleanup_after=True,
                conversation_ids=conv_ids if self.num_conversations < 10 else None,
            )
            
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
        """Generate a comprehensive report with insights and recommendations"""
        if not self.results:
            return "No results to report."
        
        # Sort by score
        sorted_results = sorted(self.results, key=lambda r: r.score(), reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]
        baseline = next((r for r in self.results if r.config.name == "baseline"), sorted_results[-1])
        
        report = []
        report.append("=" * 70)
        report.append("📊 EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 70)
        report.append(f"Run ID: {self.run_id}")
        report.append(f"Total experiments: {len(self.results)}")
        report.append(f"Conversations tested: {self.num_conversations}")
        report.append("")
        
        # === RANKINGS TABLE ===
        report.append("🏆 CONFIGURATION RANKINGS")
        report.append("-" * 70)
        report.append(f"{'Rank':<5} {'Config':<20} {'Score':<8} {'Accuracy':<10} {'Multi-hop':<10} {'Temporal':<10}")
        report.append("-" * 70)
        for i, result in enumerate(sorted_results):
            report.append(
                f"{i+1:<5} {result.config.name:<20} "
                f"{result.score():<8.4f} {result.overall_accuracy:<10.2%} "
                f"{result.multi_hop_accuracy:<10.2%} {result.temporal_accuracy:<10.2%}"
            )
        
        # === KEY INSIGHTS ===
        report.append("\n" + "=" * 70)
        report.append("💡 KEY INSIGHTS")
        report.append("=" * 70)
        
        # Improvement over baseline
        if baseline:
            improvement = best.overall_accuracy - baseline.overall_accuracy
            report.append(f"\n1. Best config improves over baseline by: {improvement:+.2%}")
            if improvement > 0.05:
                report.append("   → Significant improvement detected!")
            elif improvement < -0.02:
                report.append("   → ⚠️ Best config is WORSE than baseline - investigate!")
            else:
                report.append("   → Marginal difference - may not be statistically significant")
        
        # Multi-hop analysis (our weakness)
        multi_hop_best = max(self.results, key=lambda r: r.multi_hop_accuracy)
        multi_hop_worst = min(self.results, key=lambda r: r.multi_hop_accuracy)
        report.append(f"\n2. Multi-hop Reasoning (our weakness):")
        report.append(f"   Best:  {multi_hop_best.config.name} ({multi_hop_best.multi_hop_accuracy:.2%})")
        report.append(f"   Worst: {multi_hop_worst.config.name} ({multi_hop_worst.multi_hop_accuracy:.2%})")
        if multi_hop_best.config.extract_facts:
            report.append("   → Fact extraction helps multi-hop reasoning!")
        if multi_hop_best.config.graph_weight > 0.5:
            report.append("   → Higher graph weight helps multi-hop reasoning!")
        
        # Temporal analysis
        temporal_best = max(self.results, key=lambda r: r.temporal_accuracy)
        report.append(f"\n3. Temporal Understanding:")
        report.append(f"   Best: {temporal_best.config.name} ({temporal_best.temporal_accuracy:.2%})")
        
        # Response time analysis
        fastest = min(self.results, key=lambda r: r.avg_response_time_ms if r.avg_response_time_ms > 0 else float('inf'))
        slowest = max(self.results, key=lambda r: r.avg_response_time_ms)
        if fastest.avg_response_time_ms > 0:
            report.append(f"\n4. Response Times:")
            report.append(f"   Fastest: {fastest.config.name} ({fastest.avg_response_time_ms:.0f}ms)")
            report.append(f"   Slowest: {slowest.config.name} ({slowest.avg_response_time_ms:.0f}ms)")
        
        # === PARAMETER IMPACT ANALYSIS ===
        report.append("\n" + "=" * 70)
        report.append("📈 PARAMETER IMPACT ANALYSIS")
        report.append("=" * 70)
        
        # Compare embedding models
        small_embed = [r for r in self.results if "small" in r.config.embedding_model]
        large_embed = [r for r in self.results if "large" in r.config.embedding_model]
        if small_embed and large_embed:
            small_avg = sum(r.overall_accuracy for r in small_embed) / len(small_embed)
            large_avg = sum(r.overall_accuracy for r in large_embed) / len(large_embed)
            report.append(f"\nEmbedding Model Impact:")
            report.append(f"  text-embedding-3-small: {small_avg:.2%} avg accuracy")
            report.append(f"  text-embedding-3-large: {large_avg:.2%} avg accuracy")
            report.append(f"  → Large embeddings {'help' if large_avg > small_avg else 'hurt'} by {abs(large_avg - small_avg):.2%}")
        
        # Compare fact extraction
        with_facts = [r for r in self.results if r.config.extract_facts]
        without_facts = [r for r in self.results if not r.config.extract_facts]
        if with_facts and without_facts:
            facts_avg = sum(r.overall_accuracy for r in with_facts) / len(with_facts)
            no_facts_avg = sum(r.overall_accuracy for r in without_facts) / len(without_facts)
            report.append(f"\nFact Extraction Impact:")
            report.append(f"  Without facts: {no_facts_avg:.2%} avg accuracy")
            report.append(f"  With facts:    {facts_avg:.2%} avg accuracy")
            report.append(f"  → Fact extraction {'helps' if facts_avg > no_facts_avg else 'hurts'} by {abs(facts_avg - no_facts_avg):.2%}")
        
        # === RECOMMENDATIONS ===
        report.append("\n" + "=" * 70)
        report.append("🎯 RECOMMENDATIONS")
        report.append("=" * 70)
        
        recommendations = []
        
        # Based on best config
        if best.config.extract_facts and best.overall_accuracy > baseline.overall_accuracy:
            recommendations.append("✅ IMPLEMENT fact extraction - clear improvement")
        
        if best.config.embedding_model == "text-embedding-3-large":
            recommendations.append("✅ Use large embeddings - worth the cost")
        elif "large_embeddings" in [r.config.name for r in sorted_results[:3]]:
            recommendations.append("⚠️ Consider large embeddings - top 3 performer")
        
        if best.config.graph_weight > 0.5:
            recommendations.append(f"✅ Increase graph weight to {best.config.graph_weight}")
        
        if best.config.recall_limit > 10:
            recommendations.append(f"✅ Increase recall limit to {best.config.recall_limit}")
        
        # Warnings
        if best.overall_accuracy < 0.40:
            recommendations.append("⚠️ Overall accuracy still low - consider deeper changes")
        
        if best.multi_hop_accuracy < 0.20:
            recommendations.append("⚠️ Multi-hop still weak - need statement-level extraction")
        
        if not recommendations:
            recommendations.append("→ Baseline config is optimal for tested parameters")
        
        for rec in recommendations:
            report.append(f"  {rec}")
        
        # === BEST CONFIG DETAILS ===
        report.append("\n" + "=" * 70)
        report.append(f"🏅 RECOMMENDED CONFIG: {best.config.name}")
        report.append("=" * 70)
        report.append(f"embedding_model: {best.config.embedding_model}")
        report.append(f"enrichment_model: {best.config.enrichment_model}")
        report.append(f"recall_strategy: {best.config.recall_strategy}")
        report.append(f"recall_limit: {best.config.recall_limit}")
        report.append(f"vector_weight: {best.config.vector_weight}")
        report.append(f"graph_weight: {best.config.graph_weight}")
        report.append(f"extract_facts: {best.config.extract_facts}")
        report.append(f"graph_expansion_depth: {best.config.graph_expansion_depth}")
        
        # === NEXT STEPS ===
        report.append("\n" + "=" * 70)
        report.append("📋 SUGGESTED NEXT STEPS")
        report.append("=" * 70)
        
        if len(self.results) < 8:
            report.append("1. Run full grid search to test more configurations")
        
        if self.num_conversations < 10:
            report.append("2. Validate best config with full benchmark (10 conversations)")
        
        if best.multi_hop_accuracy < 0.30:
            report.append("3. Implement statement-level fact extraction (CORE's approach)")
        
        if best.temporal_accuracy < 0.30:
            report.append("4. Add temporal provenance to memory storage")
        
        report.append("5. Run ablation study on most impactful parameter")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.results_dir / f"{self.run_id}_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        # Also save as markdown
        md_path = self.results_dir / f"{self.run_id}_report.md"
        with open(md_path, "w") as f:
            f.write(f"# Experiment Report: {self.run_id}\n\n")
            f.write("```\n")
            f.write(report_text)
            f.write("\n```\n")
        
        return report_text


async def main():
    parser = argparse.ArgumentParser(
        description="AutoMem Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick mode (1 conv, ~5 min) - for smoke testing
  python experiment_runner.py --mode grid --quick

  # Medium mode (3 convs, ~15 min) - RECOMMENDED for iteration
  python experiment_runner.py --mode grid --medium

  # Full benchmark (10 convs, ~55 min) - for publication
  python experiment_runner.py --mode grid

  # Custom number of conversations
  python experiment_runner.py --mode grid --conversations 5

  # AI-guided exploration with medium mode
  python experiment_runner.py --mode explore --medium --iterations 10

  # Ablation study
  python experiment_runner.py --mode ablation --param recall_limit --values 5,10,15,20 --medium

Sample sizes:
  --quick:   ~200 questions, detects ~7-10% differences
  --medium:  ~600 questions, detects ~4-5% differences (recommended)
  --full:    ~2000 questions, detects ~2-3% differences
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
        help="Quick mode: 1 conversation (~200 questions, ~5 min)"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium mode: 3 conversations (~600 questions, ~15 min) - recommended for iteration"
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="Explicit number of conversations to test (1-10)"
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
    
    # Determine number of conversations
    if args.conversations:
        num_convs = args.conversations
    elif args.quick:
        num_convs = 1
    elif args.medium:
        num_convs = 3
    else:
        num_convs = 10
    
    print(f"📊 Mode: {num_convs} conversation(s) (~{num_convs * 200} questions)")
    
    # Initialize runner
    runner = ExperimentRunner(
        results_dir=args.output_dir,
        use_railway=args.railway,
        max_parallel=args.parallel,
        num_conversations=num_convs,
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

