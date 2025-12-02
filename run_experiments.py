"""
Comprehensive Experiment Runner

This script runs all experiments for the AI Contraception Counseling System.
It provides options to run individual experiments or all experiments sequentially.

Usage:
    # Run all experiments
    python run_experiments.py --all

    # Run specific experiment
    python run_experiments.py --exp 1

    # Run multiple specific experiments
    python run_experiments.py --exp 1 2 3

    # Skip specific experiments
    python run_experiments.py --all --skip 5

    # Dry run (validate without executing)
    python run_experiments.py --dry-run
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'results/logs/experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the execution of all experiments."""

    EXPERIMENTS = {
        1: {
            'name': 'Baseline Knowledge Test',
            'script': 'experiments/exp1_baseline_knowledge.py',
            'description': 'Tests LLM baseline knowledge without RAG',
            'estimated_time': '10-15 min',
            'dependencies': ['data/synthetic/qa_pairs.json']
        },
        2: {
            'name': 'Anchored Prompts',
            'script': 'experiments/exp2_anchored_prompts.py',
            'description': 'Evaluates strict guideline-following prompts',
            'estimated_time': '10-15 min',
            'dependencies': ['data/synthetic/qa_pairs.json']
        },
        3: {
            'name': 'RAG Comparison',
            'script': 'experiments/exp3_rag_comparison.py',
            'description': 'Compares RAG vs non-RAG performance',
            'estimated_time': '15-20 min',
            'dependencies': ['data/synthetic/qa_pairs.json', 'data/processed/vector_store/']
        },
        4: {
            'name': 'Long Session Forgetting (4a)',
            'script': 'experiments/exp4a_long_session_forgetting.py',
            'description': 'Tests conversation memory across long sessions',
            'estimated_time': '20-30 min',
            'dependencies': ['data/synthetic/long_session_conversations.json']
        },
        5: {
            'name': 'Multi-Session Memory (4b)',
            'script': 'experiments/exp4b_multi_session_memory.py',
            'description': 'Tests memory across multiple sessions',
            'estimated_time': '20-30 min',
            'dependencies': ['data/synthetic/multi_session_scenarios.json']
        },
        6: {
            'name': 'Adherence RL',
            'script': 'experiments/exp5_adherence_rl.py',
            'description': 'Tests reinforcement learning for adherence',
            'estimated_time': '30-60 min',
            'dependencies': ['data/synthetic/adherence_dataset.json']
        }
    }

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.results_dir / "tables").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "evaluation").mkdir(exist_ok=True)

        self.run_summary = {
            'start_time': None,
            'end_time': None,
            'experiments_run': [],
            'experiments_failed': [],
            'total_duration': 0
        }

    def check_dependencies(self, exp_num: int) -> tuple[bool, List[str]]:
        """Check if all dependencies for an experiment exist."""
        exp = self.EXPERIMENTS[exp_num]
        missing = []

        for dep in exp['dependencies']:
            dep_path = Path(dep)
            if not dep_path.exists():
                missing.append(dep)

        return len(missing) == 0, missing

    def validate_environment(self) -> bool:
        """Validate that the environment is ready for experiments."""
        logger.info("=" * 80)
        logger.info("VALIDATING ENVIRONMENT")
        logger.info("=" * 80)

        all_valid = True

        # Check Python imports
        logger.info("\n1. Checking Python imports...")
        required_modules = [
            'src.rag.rag_pipeline',
            'src.evaluation.metrics',
            'src.memory.memory_manager',
            'ollama'
        ]

        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"   [OK] {module}")
            except ImportError as e:
                logger.error(f"   [FAIL] {module}: {e}")
                all_valid = False

        # Check Ollama models
        logger.info("\n2. Checking Ollama models...")
        try:
            import ollama
            models_response = ollama.list()
            required_models = ['llama3.2', 'aya:8b']

            # Handle different response types
            if hasattr(models_response, 'models'):
                model_list = models_response.models
            elif isinstance(models_response, dict):
                model_list = models_response.get('models', [])
            else:
                model_list = []

            # Extract model names
            model_names = []
            for m in model_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict):
                    model_names.append(m.get('name', m.get('model', '')))

            for model in required_models:
                # Check if model name starts with required name
                found = any(mn.startswith(model) or mn.split(':')[0] == model for mn in model_names)
                if found:
                    logger.info(f"   [OK] {model}")
                else:
                    logger.warning(f"   [WARN] {model} not found")
                    if model == 'aya:8b':
                        logger.info("      (Aya is optional - llama3.2 will be used for all languages)")
        except Exception as e:
            logger.error(f"   [FAIL] Cannot connect to Ollama: {e}")
            all_valid = False

        # Check data files
        logger.info("\n3. Checking data files...")
        critical_files = [
            'data/synthetic/eval_questions.json',
            'data/synthetic/qa_pairs.json',
            'data/synthetic/multilang_qa_pairs.json',
            'data/processed/vector_store/faiss.index',
            'data/processed/vector_store/chunks.json'
        ]

        for file in critical_files:
            file_path = Path(file)
            if file_path.exists():
                logger.info(f"   [OK] {file}")
            else:
                logger.error(f"   [FAIL] {file} not found")
                all_valid = False

        # Check config
        logger.info("\n4. Checking configuration...")
        config_path = Path('configs/config.yaml')
        if config_path.exists():
            logger.info(f"   [OK] config.yaml")
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                logger.info(f"   [OK] Config valid YAML")
            except Exception as e:
                logger.error(f"   [FAIL] Config parse error: {e}")
                all_valid = False
        else:
            logger.error(f"   [FAIL] config.yaml not found")
            all_valid = False

        logger.info("\n" + "=" * 80)
        if all_valid:
            logger.info("[PASS] ENVIRONMENT VALIDATION PASSED")
        else:
            logger.error("[FAIL] ENVIRONMENT VALIDATION FAILED")
        logger.info("=" * 80 + "\n")

        return all_valid

    def run_experiment(self, exp_num: int, dry_run: bool = False) -> Dict:
        """Run a single experiment."""
        if exp_num not in self.EXPERIMENTS:
            raise ValueError(f"Experiment {exp_num} does not exist")

        exp = self.EXPERIMENTS[exp_num]

        logger.info("=" * 80)
        logger.info(f"EXPERIMENT {exp_num}: {exp['name']}")
        logger.info("=" * 80)
        logger.info(f"Description: {exp['description']}")
        logger.info(f"Estimated time: {exp['estimated_time']}")
        logger.info(f"Script: {exp['script']}")

        # Check dependencies
        logger.info("\nChecking dependencies...")
        deps_ok, missing = self.check_dependencies(exp_num)

        if not deps_ok:
            logger.error(f"❌ Missing dependencies:")
            for dep in missing:
                logger.error(f"   - {dep}")
            return {
                'experiment': exp_num,
                'name': exp['name'],
                'status': 'failed',
                'reason': 'missing_dependencies',
                'missing': missing
            }

        logger.info("[OK] All dependencies found")

        if dry_run:
            logger.info("\n[DRY-RUN] Would execute experiment but --dry-run flag is set")
            return {
                'experiment': exp_num,
                'name': exp['name'],
                'status': 'dry_run',
                'duration': 0
            }

        # Run the experiment
        logger.info(f"\n[RUNNING] Starting experiment...")
        start_time = time.time()

        try:
            # Import and run the experiment
            import importlib.util
            spec = importlib.util.spec_from_file_location("experiment", exp['script'])
            module = importlib.util.module_from_spec(spec)
            sys.modules["experiment"] = module
            spec.loader.exec_module(module)

            # Execute main function
            if hasattr(module, 'main'):
                module.main()
            else:
                logger.warning("No main() function found in experiment script")

            duration = time.time() - start_time

            logger.info(f"\n[SUCCESS] Experiment {exp_num} completed successfully")
            logger.info(f"[TIME] Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")

            return {
                'experiment': exp_num,
                'name': exp['name'],
                'status': 'success',
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"\n[FAIL] Experiment {exp_num} failed after {duration:.2f} seconds")
            logger.error(f"Error: {str(e)}")

            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")

            return {
                'experiment': exp_num,
                'name': exp['name'],
                'status': 'failed',
                'reason': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }

    def run_all(self, skip: Optional[List[int]] = None, dry_run: bool = False):
        """Run all experiments sequentially."""
        skip = skip or []

        logger.info("=" * 80)
        logger.info("RUNNING ALL EXPERIMENTS")
        logger.info("=" * 80)

        experiments_to_run = [exp for exp in self.EXPERIMENTS.keys() if exp not in skip]

        logger.info(f"\nExperiments to run: {experiments_to_run}")
        if skip:
            logger.info(f"Skipping: {skip}")

        total_estimated_time = sum(
            int(self.EXPERIMENTS[exp]['estimated_time'].split('-')[0])
            for exp in experiments_to_run
        )
        logger.info(f"\nEstimated total time: ~{total_estimated_time}-{total_estimated_time*1.5:.0f} minutes")

        if not dry_run:
            response = input("\nProceed with experiment execution? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("Experiment execution cancelled by user")
                return

        self.run_summary['start_time'] = datetime.now().isoformat()

        for exp_num in experiments_to_run:
            result = self.run_experiment(exp_num, dry_run=dry_run)

            if result['status'] == 'success':
                self.run_summary['experiments_run'].append(result)
            elif result['status'] == 'failed':
                self.run_summary['experiments_failed'].append(result)

            # Save intermediate results
            self._save_run_summary()

            logger.info("\n" + "=" * 80 + "\n")

        self.run_summary['end_time'] = datetime.now().isoformat()

        if not dry_run:
            self.run_summary['total_duration'] = sum(
                r['duration'] for r in self.run_summary['experiments_run']
            )

        self._print_final_summary()
        self._save_run_summary()

    def _print_final_summary(self):
        """Print final summary of all experiments."""
        logger.info("=" * 80)
        logger.info("EXPERIMENT RUN SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nStart time: {self.run_summary['start_time']}")
        logger.info(f"End time: {self.run_summary['end_time']}")

        total_experiments = len(self.run_summary['experiments_run']) + len(self.run_summary['experiments_failed'])
        logger.info(f"\nTotal experiments: {total_experiments}")
        logger.info(f"[OK] Successful: {len(self.run_summary['experiments_run'])}")
        logger.info(f"[FAIL] Failed: {len(self.run_summary['experiments_failed'])}")

        if self.run_summary['total_duration'] > 0:
            logger.info(f"\nTotal duration: {self.run_summary['total_duration']:.2f} seconds")
            logger.info(f"                ({self.run_summary['total_duration']/60:.1f} minutes)")

        if self.run_summary['experiments_run']:
            logger.info("\n[SUCCESS] Successful Experiments:")
            for result in self.run_summary['experiments_run']:
                logger.info(f"   {result['experiment']}. {result['name']} ({result['duration']:.1f}s)")

        if self.run_summary['experiments_failed']:
            logger.info("\n[FAIL] Failed Experiments:")
            for result in self.run_summary['experiments_failed']:
                logger.info(f"   {result['experiment']}. {result['name']}: {result.get('reason', 'Unknown error')}")

        logger.info("\n" + "=" * 80)

    def _save_run_summary(self):
        """Save run summary to file."""
        summary_file = self.results_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.run_summary, f, indent=2)

        logger.info(f"\n Run summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for AI Contraception Counseling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_experiments.py --all

  # Run specific experiments
  python run_experiments.py --exp 1 2 3

  # Run all except experiment 5
  python run_experiments.py --all --skip 5

  # Validate environment without running
  python run_experiments.py --validate

  # Dry run (check without executing)
  python run_experiments.py --all --dry-run
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )

    parser.add_argument(
        '--exp',
        type=int,
        nargs='+',
        help='Run specific experiment(s) by number (1-6)'
    )

    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        help='Skip specific experiment(s) when using --all'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate experiments without running them'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate environment and exit'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )

    args = parser.parse_args()

    runner = ExperimentRunner()

    # List experiments
    if args.list:
        print("\n" + "=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80 + "\n")

        for exp_num, exp in runner.EXPERIMENTS.items():
            print(f"{exp_num}. {exp['name']}")
            print(f"   Description: {exp['description']}")
            print(f"   Estimated time: {exp['estimated_time']}")
            print(f"   Script: {exp['script']}\n")

        return

    # Validate environment
    if args.validate or args.dry_run:
        if not runner.validate_environment():
            sys.exit(1)

        if args.validate:
            return

    # Run experiments
    if args.all:
        runner.run_all(skip=args.skip, dry_run=args.dry_run)
    elif args.exp:
        # Validate first
        if not runner.validate_environment():
            response = input("\nEnvironment validation failed. Continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                sys.exit(1)

        runner.run_summary['start_time'] = datetime.now().isoformat()

        for exp_num in args.exp:
            result = runner.run_experiment(exp_num, dry_run=args.dry_run)

            if result['status'] == 'success':
                runner.run_summary['experiments_run'].append(result)
            elif result['status'] == 'failed':
                runner.run_summary['experiments_failed'].append(result)

        runner.run_summary['end_time'] = datetime.now().isoformat()
        runner._print_final_summary()
        runner._save_run_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
