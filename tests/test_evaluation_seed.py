import unittest

from src.baselines.strategies import FixedPriceStrategy
from src.environment.pricing_env import PricingEnvironment
from src.evaluation.comparison import _run_strategy_episode
from src.evaluation.evaluate import _run_greedy_episode
from src.evaluation.eval_summary_export import build_eval_summary


class FixedPolicy:
    def select_action(self, state, epsilon=0.0):
        return 2


class EvaluationSeedTests(unittest.TestCase):
    def test_exported_summary_preserves_seed_metadata(self):
        summary = build_eval_summary({"seed": 73, "avg_reward": 1.5})

        self.assertEqual(summary["seed"], 73)

    def test_legacy_summary_without_seed_uses_null_metadata(self):
        summary = build_eval_summary({"avg_reward": 1.5})

        self.assertIsNone(summary["seed"])

    def test_greedy_evaluation_repeats_with_same_episode_seed(self):
        env = PricingEnvironment(horizon=20, initial_inventory=100, demand_noise=10)
        agent = FixedPolicy()

        first = _run_greedy_episode(agent, env, max_steps=20, seed=17)
        second = _run_greedy_episode(agent, env, max_steps=20, seed=17)

        self.assertEqual(first, second)

    def test_comparison_episode_repeats_with_same_episode_seed(self):
        env = PricingEnvironment(horizon=20, initial_inventory=100, demand_noise=10)
        strategy = FixedPriceStrategy(action=2)

        first = _run_strategy_episode(strategy, env, max_steps=20, seed=29)
        second = _run_strategy_episode(strategy, env, max_steps=20, seed=29)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
