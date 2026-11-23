import unittest

from src.baselines.strategies import FixedPriceStrategy
from src.environment.pricing_env import PricingEnvironment
from src.evaluation.comparison import _aggregate_kpis, _run_strategy_episode


class ComparisonKpiTests(unittest.TestCase):
    def test_comparison_uses_environment_revenue_and_caps_inventory(self):
        env = PricingEnvironment(horizon=5, initial_inventory=1, demand_noise=0)
        episode = _run_strategy_episode(
            FixedPriceStrategy(action=2),
            env,
            max_steps=5,
        )

        self.assertEqual(episode["total_bookings"], 1)
        self.assertEqual(episode["total_revenue"], 55.0)
        self.assertEqual(episode["total_reward"], 55.0)
        self.assertEqual(episode["steps"], 1)

        kpis = _aggregate_kpis([episode], inventory_per_episode=1)

        self.assertEqual(kpis["occupancy_rate"], 1.0)
        self.assertEqual(kpis["avg_selling_price"], 55.0)
        self.assertEqual(kpis["avg_reward"], kpis["total_revenue"])


if __name__ == "__main__":
    unittest.main()
