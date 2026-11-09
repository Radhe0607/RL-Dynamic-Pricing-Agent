import unittest

from src.config.config_loader import (
    ConfigValidationError,
    flatten_config,
    merge_with_defaults,
    validate_config,
)


class SeedConfigTests(unittest.TestCase):
    def test_zero_and_maximum_seed_are_valid(self):
        for seed in (0, 2**32 - 1):
            with self.subTest(seed=seed):
                values = flatten_config({"training": {"seed": seed}})
                validate_config(values)
                self.assertEqual(merge_with_defaults(values).seed, seed)

    def test_negative_seed_is_rejected_with_config_error(self):
        values = flatten_config({"training": {"seed": -1}})
        with self.assertRaises(ConfigValidationError):
            validate_config(values)

    def test_seed_above_supported_rng_range_is_rejected_with_config_error(self):
        values = flatten_config({"training": {"seed": 2**32}})
        with self.assertRaisesRegex(ConfigValidationError, "0 to 2\\*\\*32 - 1"):
            validate_config(values)


if __name__ == "__main__":
    unittest.main()
