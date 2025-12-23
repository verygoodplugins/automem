#!/usr/bin/env python3
"""
AutoMem Analysis Agent

An LLM-powered agent that analyzes experiment results and decides what to try next.
Runs autonomously until it finds an optimal configuration or hits iteration limit.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_config import ExperimentConfig, ExperimentResult
from experiment_runner import ExperimentRunner
from openai import AsyncOpenAI

ANALYSIS_PROMPT = """You are an AI agent optimizing a memory retrieval system for conversational AI.

## Current Results
{results_summary}

## Goal
Find the configuration that maximizes:
1. Overall accuracy (weight: 40%)
2. Multi-hop reasoning (weight: 20%) - this is our weakest category
3. Temporal understanding (weight: 15%)
4. Fast response times (weight: 15%) - target < 100ms
5. Cost efficiency (weight: 10%)

## Constraints
- Response time must stay under 200ms
- Must work with production traffic patterns
- Budget: ~$0.10 per 1000 queries

## Your Task
Based on the results so far, decide:
1. Should we continue exploring? (yes/no)
2. If yes, what specific configuration changes should we try next?
3. What hypothesis are you testing with these changes?

Respond in JSON format:
{{
    "continue_exploring": true/false,
    "reasoning": "Your analysis of the results...",
    "next_experiments": [
        {{
            "name": "experiment_name",
            "changes": {{"param": "value", ...}},
            "hypothesis": "What you expect to learn..."
        }}
    ],
    "confidence": 0.0-1.0,  // How confident are you in your recommendations?
    "best_config_so_far": "name of best configuration"
}}
"""


class AnalysisAgent:
    """
    LLM-powered agent for autonomous experiment optimization.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        results_dir: str = "experiment_results",
        max_iterations: int = 10,
        min_improvement_threshold: float = 0.02,
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.results_dir = Path(results_dir)
        self.max_iterations = max_iterations
        self.min_improvement_threshold = min_improvement_threshold
        self.history: List[Dict[str, Any]] = []
        self._pending_configs: List[ExperimentConfig] = []

    def _format_results(self, results: List[ExperimentResult]) -> str:
        """Format results for the LLM"""
        lines = []
        for r in results:
            lines.append(
                f"""
### {r.config.name}
- Overall Accuracy: {r.overall_accuracy:.2%}
- Single-hop: {r.single_hop_accuracy:.2%}
- Multi-hop: {r.multi_hop_accuracy:.2%}
- Temporal: {r.temporal_accuracy:.2%}
- Open Domain: {r.open_domain_accuracy:.2%}
- Adversarial: {r.adversarial_accuracy:.2%}
- Response Time: {r.avg_response_time_ms:.0f}ms
- Score: {r.score():.4f}

Config:
- Embedding: {r.config.embedding_model}
- Enrichment: {r.config.enrichment_model}
- Recall Limit: {r.config.recall_limit}
- Vector Weight: {r.config.vector_weight}
- Graph Weight: {r.config.graph_weight}
- Extract Facts: {r.config.extract_facts}
"""
            )
        return "\n".join(lines)

    async def analyze_and_decide(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Analyze results and decide next steps.
        Returns decision dict with continue_exploring, next_experiments, etc.
        """
        results_summary = self._format_results(results)

        # Add history context
        if self.history:
            history_summary = "\n## Previous Iterations\n"
            for h in self.history[-3:]:  # Last 3 iterations
                history_summary += (
                    f"- Iteration {h['iteration']}: Best score {h['best_score']:.4f}\n"
                )
            results_summary = history_summary + "\n" + results_summary

        prompt = ANALYSIS_PROMPT.format(results_summary=results_summary)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in machine learning optimization and memory systems.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            decision = json.loads(response.choices[0].message.content)
            return decision

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {
                "continue_exploring": False,
                "reasoning": f"Analysis error: {e}",
                "next_experiments": [],
                "confidence": 0.0,
            }

    def _create_config_from_changes(
        self,
        base_config: ExperimentConfig,
        changes: Dict[str, Any],
        name: str,
    ) -> ExperimentConfig:
        """Create a new config by applying changes to base"""
        config_dict = base_config.__dict__.copy()
        config_dict["name"] = name

        # Map common LLM naming variations to our field names
        key_mapping = {
            "embedding": "embedding_model",
            "enrichment": "enrichment_model",
            "recall": "recall_limit",
            "vector": "vector_weight",
            "graph": "graph_weight",
            "facts": "extract_facts",
        }

        # Normalize and apply changes
        for key, value in changes.items():
            # Normalize key (lowercase, remove spaces)
            norm_key = key.lower().replace(" ", "_").replace("-", "_")

            # Check for common mappings
            if norm_key in key_mapping:
                norm_key = key_mapping[norm_key]

            # Only update if it's a valid config field
            if norm_key in config_dict:
                config_dict[norm_key] = value

        return ExperimentConfig(**config_dict)

    async def run_autonomous_optimization(
        self,
        initial_configs: Optional[List[ExperimentConfig]] = None,
        runner: Optional[ExperimentRunner] = None,
    ) -> ExperimentConfig:
        """
        Run autonomous optimization loop.

        The agent will:
        1. Run experiments
        2. Analyze results
        3. Decide what to try next
        4. Repeat until convergence or max iterations
        """
        if runner is None:
            runner = ExperimentRunner(
                results_dir=str(self.results_dir),
                questions_per_config=200,  # Quick tests
            )

        if initial_configs is None:
            from experiment_config import generate_experiment_grid

            initial_configs = generate_experiment_grid()[:3]  # Start with 3

        all_results: List[ExperimentResult] = []
        best_score = 0.0
        best_config = initial_configs[0]
        iteration = 0

        print("\n" + "=" * 70)
        print("🤖 AUTONOMOUS OPTIMIZATION AGENT STARTED")
        print("=" * 70)

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"📍 ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            # Run experiments
            if iteration == 1:
                configs_to_run = initial_configs
            else:
                # Get configs from last decision
                configs_to_run = self._pending_configs

            print(f"🔬 Running {len(configs_to_run)} experiments...")
            results = await runner.run_grid_search(configs_to_run)
            all_results.extend(results)

            # Find current best
            if results:
                current_best = max(results, key=lambda r: r.score())
                current_best_score = current_best.score()

                if current_best_score > best_score:
                    improvement = current_best_score - best_score
                    best_score = current_best_score
                    best_config = current_best.config
                    print(f"✨ New best! {best_config.name}: {best_score:.4f} (+{improvement:.4f})")
                else:
                    print(f"📊 Current best still: {best_config.name}: {best_score:.4f}")

            # Record history
            self.history.append(
                {
                    "iteration": iteration,
                    "best_score": best_score,
                    "best_config": best_config.name,
                    "experiments_run": len(configs_to_run),
                }
            )

            # Ask the agent what to do next
            print(f"\n🤔 Analyzing results...")
            decision = await self.analyze_and_decide(all_results)

            print(f"\n📋 Agent Decision:")
            print(f"   Continue: {decision.get('continue_exploring', False)}")
            print(f"   Confidence: {decision.get('confidence', 0):.2f}")
            print(f"   Reasoning: {decision.get('reasoning', 'N/A')[:200]}...")

            if not decision.get("continue_exploring", False):
                print(f"\n🎯 Agent decided to stop. Reason: {decision.get('reasoning', 'N/A')}")
                break

            # Create configs for next iteration
            next_experiments = decision.get("next_experiments", [])
            if not next_experiments:
                print("⚠️ No next experiments suggested, stopping.")
                break

            self._pending_configs = []
            for exp in next_experiments:
                config = self._create_config_from_changes(
                    best_config, exp.get("changes", {}), exp.get("name", f"agent_exp_{iteration}")
                )
                self._pending_configs.append(config)
                print(f"   📝 Will try: {config.name}")
                print(f"      Hypothesis: {exp.get('hypothesis', 'N/A')}")

        # Final report
        print("\n" + "=" * 70)
        print("🏆 OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Best Configuration: {best_config.name}")
        print(f"Best Score: {best_score:.4f}")
        print(f"Iterations: {iteration}")
        print(f"\nFinal Config:")
        print(best_config.to_json())

        # Save final report
        self._save_final_report(best_config, best_score, all_results)

        return best_config

    def _save_final_report(
        self,
        best_config: ExperimentConfig,
        best_score: float,
        all_results: List[ExperimentResult],
    ):
        """Save comprehensive final report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "best_config": best_config.__dict__,
            "best_score": best_score,
            "iterations": len(self.history),
            "history": self.history,
            "all_experiments": [
                {
                    "name": r.config.name,
                    "score": r.score(),
                    "accuracy": r.overall_accuracy,
                    "multi_hop": r.multi_hop_accuracy,
                    "temporal": r.temporal_accuracy,
                    "response_time_ms": r.avg_response_time_ms,
                }
                for r in all_results
            ],
        }

        report_path = (
            self.results_dir
            / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Report saved to: {report_path}")


async def main():
    """Run autonomous optimization"""
    import argparse

    parser = argparse.ArgumentParser(description="AutoMem Autonomous Optimization Agent")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="LLM model for analysis")
    parser.add_argument(
        "--output-dir", type=str, default="experiment_results", help="Results directory"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer questions")

    args = parser.parse_args()

    agent = AnalysisAgent(
        model=args.model,
        results_dir=args.output_dir,
        max_iterations=args.iterations,
    )

    runner = ExperimentRunner(
        results_dir=args.output_dir,
        questions_per_config=100 if args.quick else None,
    )

    best_config = await agent.run_autonomous_optimization(runner=runner)

    print("\n✅ Optimization complete!")
    print(f"Best configuration saved to: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
