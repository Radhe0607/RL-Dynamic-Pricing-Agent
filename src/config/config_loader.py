"""
config_loader.py
----------------
External Configuration File Loader for the DQN Dynamic Pricing Agent.

This module reads a JSON configuration file, validates every value, merges
it with the ``DQNConfig`` dataclass defaults, and returns a fully populated
``DQNConfig`` instance ready for use by the training loop.

Design philosophy
-----------------
* **Additive override** — the JSON file only needs to contain the keys the
  user wants to change.  Any key absent from the file falls back to its
  ``DQNConfig`` default.  This prevents the file from becoming a
  maintenance burden when new hyperparameters are added.

* **Early failure** — all validation runs before training starts, so the
  user sees a clear error message rather than a cryptic crash 200 episodes
  into a run.

* **No external dependencies** — the standard-library ``json`` module is
  used exclusively.  No ``PyYAML``, ``pydantic``, or ``hydra`` needed.

JSON file structure
-------------------
The config file is organised into named sections::

    {
      "environment":   { "state_size": 3, "action_size": 5, ... },
      "training":      { "max_episodes": 1000, "save_every": 100 },
      "replay_buffer": { "buffer_capacity": 10000, "batch_size": 64, ... },
      "network":       { "hidden_size": 64 },
      "optimiser":     { "learning_rate": 0.001 },
      "rl_core":       { "gamma": 0.99, "tau": 0.005, ... },
      "exploration":   { "epsilon_start": 1.0, "epsilon_end": 0.05, ... },
      "checkpointing": { "checkpoint_dir": "checkpoints" }
    }

Keys starting with ``"_"`` (e.g. ``"_comment"``) are silently ignored so
the file can contain human-readable annotations.

Validation rules
-----------------
  learning_rate    : float  in (0, 1]
  gamma            : float  in (0, 1]
  tau              : float  in (0, 1]
  epsilon_start    : float  in (0, 1]
  epsilon_end      : float  in (0, 1)   and < epsilon_start
  epsilon_decay    : float  in (0, 1)
  batch_size       : int    ≥ 1
  buffer_capacity  : int    ≥ batch_size
  min_buffer_size  : int    ≥ batch_size
  hidden_size      : int    ≥ 1
  max_episodes     : int    ≥ 1
  max_steps_per_episode : int ≥ 1
  state_size       : int    ≥ 1
  action_size      : int    ≥ 2
  target_update_freq : int  ≥ 1
  save_every       : int    ≥ 1
  checkpoint_dir   : str    (non-empty)

Public API
----------
    load_config(config_path) → DQNConfig
        Main entry point.  Load, validate, merge, return DQNConfig.

    validate_config(flat_dict) → None
        Raises ``ConfigValidationError`` on the first invalid value found.

    merge_with_defaults(flat_dict) → DQNConfig
        Produce a DQNConfig by merging flat_dict over default values.

    flatten_config(nested_dict) → dict
        Collapse the nested JSON sections into a flat key → value mapping.

Exceptions
----------
    ConfigValidationError(ValueError)
        Raised with a descriptive message whenever a value fails validation.
        Includes the key name, the bad value, and a human-readable rule.

    ConfigFileError(FileNotFoundError / json.JSONDecodeError)
        Raised when the config file is missing or contains malformed JSON.

Usage
-----
    from src.config.config_loader import load_config

    cfg = load_config("config/training_config.json")
    # cfg is a fully validated DQNConfig instance
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.config.dqn_config import DQNConfig


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Raised when a hyperparameter value fails a validation rule.

    Args:
        key   (str): The parameter name that failed validation.
        value (Any): The invalid value that was provided.
        rule  (str): Human-readable description of the constraint.
    """

    def __init__(self, key: str, value: Any, rule: str) -> None:
        self.key   = key
        self.value = value
        self.rule  = rule
        super().__init__(
            f"Invalid config value for '{key}': got {value!r}. "
            f"Rule: {rule}"
        )


class ConfigFileError(OSError):
    """Raised when the configuration file cannot be read or parsed.

    Args:
        path   (str): Path of the file that caused the error.
        reason (str): Human-readable description of the failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path   = path
        self.reason = reason
        super().__init__(f"Cannot load config from '{path}': {reason}")


# ---------------------------------------------------------------------------
# Section → flat-key mapping
# ---------------------------------------------------------------------------

# Maps each JSON section name to the set of DQNConfig field names it covers.
# Any unknown key inside a section triggers a printed warning (not an error).
_SECTION_FIELDS: Dict[str, list] = {
    "environment":   ["state_size", "action_size",
                      "max_steps_per_episode"],
    "training":      ["max_episodes", "save_every"],
    "replay_buffer": ["buffer_capacity", "batch_size", "min_buffer_size"],
    "network":       ["hidden_size"],
    "optimiser":     ["learning_rate"],
    "rl_core":       ["gamma", "tau", "target_update_freq"],
    "exploration":   ["epsilon_start", "epsilon_end", "epsilon_decay"],
    "checkpointing": ["checkpoint_dir"],
}

# Invert — maps each flat key to its expected section (for error reporting)
_KEY_TO_SECTION: Dict[str, str] = {
    field: section
    for section, fields in _SECTION_FIELDS.items()
    for field in fields
}


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

def _check_positive_float(key: str, value: Any, upper: float = 1.0) -> float:
    """Assert *value* is a float in ``(0, upper]``.

    Args:
        key   (str):   Parameter name (for the error message).
        value (Any):   Value to validate.
        upper (float): Upper bound (inclusive). Default 1.0.

    Returns:
        float: The validated float value.

    Raises:
        ConfigValidationError: If *value* fails the constraint.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ConfigValidationError(key, value, "must be a number")
    if not (0.0 < v <= upper):
        raise ConfigValidationError(
            key, v, f"must be a float in (0, {upper}]"
        )
    return v


def _check_positive_int(key: str, value: Any, minimum: int = 1) -> int:
    """Assert *value* is an integer ≥ *minimum*.

    Args:
        key     (str): Parameter name (for the error message).
        value   (Any): Value to validate.
        minimum (int): Lower bound (inclusive). Default 1.

    Returns:
        int: The validated integer value.

    Raises:
        ConfigValidationError: If *value* fails the constraint.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ConfigValidationError(key, value, "must be an integer")
    if v < minimum:
        raise ConfigValidationError(
            key, v, f"must be an integer ≥ {minimum}"
        )
    return v


def validate_config(flat: Dict[str, Any]) -> None:
    """Validate every hyperparameter in a flat key→value dictionary.

    Checks are performed in a logical order so the first failure
    surfaces the most actionable error.  Call this *before* constructing
    ``DQNConfig`` to get clear, early feedback.

    Args:
        flat (dict): Flat key→value mapping (output of ``flatten_config``).

    Raises:
        ConfigValidationError: On the first invalid value found.
    """

    # Helper — validate only if the key is present (absent keys use defaults)
    def _get(key: str, default: Any = None) -> Any:
        return flat.get(key, default)

    # ── Float parameters ──────────────────────────────────────────────────
    if "learning_rate" in flat:
        _check_positive_float("learning_rate", flat["learning_rate"], upper=1.0)

    if "gamma" in flat:
        _check_positive_float("gamma", flat["gamma"], upper=1.0)

    if "tau" in flat:
        _check_positive_float("tau", flat["tau"], upper=1.0)

    if "epsilon_decay" in flat:
        v = float(flat["epsilon_decay"])
        if not (0.0 < v < 1.0):
            raise ConfigValidationError(
                "epsilon_decay", v, "must be a float in (0, 1)"
            )

    # ── Epsilon ordering ──────────────────────────────────────────────────
    eps_start = float(_get("epsilon_start", 1.0))
    eps_end   = float(_get("epsilon_end",   0.05))

    if "epsilon_start" in flat:
        if not (0.0 < eps_start <= 1.0):
            raise ConfigValidationError(
                "epsilon_start", eps_start, "must be a float in (0, 1]"
            )

    if "epsilon_end" in flat:
        if not (0.0 < eps_end < 1.0):
            raise ConfigValidationError(
                "epsilon_end", eps_end, "must be a float in (0, 1)"
            )

    if "epsilon_start" in flat or "epsilon_end" in flat:
        if eps_end >= eps_start:
            raise ConfigValidationError(
                "epsilon_end", eps_end,
                f"must be strictly less than epsilon_start ({eps_start})"
            )

    # ── Integer parameters ────────────────────────────────────────────────
    if "batch_size" in flat:
        batch = _check_positive_int("batch_size", flat["batch_size"], minimum=1)
    else:
        batch = 64   # DQNConfig default

    if "buffer_capacity" in flat:
        cap = _check_positive_int("buffer_capacity", flat["buffer_capacity"], minimum=1)
        if cap < batch:
            raise ConfigValidationError(
                "buffer_capacity", cap,
                f"must be ≥ batch_size ({batch})"
            )

    if "min_buffer_size" in flat:
        mbuf = _check_positive_int("min_buffer_size", flat["min_buffer_size"], minimum=1)
        if mbuf < batch:
            raise ConfigValidationError(
                "min_buffer_size", mbuf,
                f"must be ≥ batch_size ({batch})"
            )

    for key in ("hidden_size", "max_episodes", "max_steps_per_episode",
                "target_update_freq", "save_every"):
        if key in flat:
            _check_positive_int(key, flat[key], minimum=1)

    if "state_size" in flat:
        _check_positive_int("state_size", flat["state_size"], minimum=1)

    if "action_size" in flat:
        _check_positive_int("action_size", flat["action_size"], minimum=2)

    # ── String parameters ─────────────────────────────────────────────────
    if "checkpoint_dir" in flat:
        val = flat["checkpoint_dir"]
        if not isinstance(val, str) or not val.strip():
            raise ConfigValidationError(
                "checkpoint_dir", val, "must be a non-empty string"
            )


# ---------------------------------------------------------------------------
# Flattening helper
# ---------------------------------------------------------------------------

def flatten_config(nested: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a nested JSON config dict into a flat key→value mapping.

    Keys starting with ``"_"`` are treated as comments and skipped.
    Values that are dicts are recursively flattened.
    Section names (keys whose values are dicts) are dropped; only
    leaf values are kept.

    Args:
        nested (dict): Nested JSON-parsed dict (as returned by ``json.load``).

    Returns:
        dict: Flat ``{param_name: value}`` mapping.

    Example::

        {
          "optimiser": {"learning_rate": 0.001},
          "rl_core":   {"gamma": 0.99}
        }
        →  {"learning_rate": 0.001, "gamma": 0.99}
    """
    flat: Dict[str, Any] = {}

    for key, value in nested.items():
        # Skip comment keys
        if key.startswith("_"):
            continue

        if isinstance(value, dict):
            # Recurse into sections; warn about unexpected leaf keys later
            sub = flatten_config(value)
            for sub_key, sub_val in sub.items():
                if sub_key in flat:
                    print(
                        f"  [config] WARNING: duplicate key '{sub_key}' "
                        f"in section '{key}' — last value wins."
                    )
                flat[sub_key] = sub_val
        else:
            flat[key] = value

    return flat


# ---------------------------------------------------------------------------
# Merge with defaults
# ---------------------------------------------------------------------------

def merge_with_defaults(flat: Dict[str, Any]) -> DQNConfig:
    """Overlay *flat* values over ``DQNConfig`` defaults and return a new instance.

    Only keys that are valid ``DQNConfig`` fields are applied.  Unknown keys
    trigger a printed warning but do not raise an exception — this keeps the
    loader lenient for forward-compatibility.

    Args:
        flat (dict): Flat key→value mapping (already validated).

    Returns:
        DQNConfig: A new config instance with overrides applied.
    """
    # Start with pure defaults
    defaults = DQNConfig()
    kwargs   = {}

    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(DQNConfig)}

    for key, value in flat.items():
        if key in valid_fields:
            # Cast to the field's declared type to be safe
            field_type = type(getattr(defaults, key))
            try:
                kwargs[key] = field_type(value)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    key, value,
                    f"cannot convert to expected type {field_type.__name__}: {exc}"
                )
        else:
            print(
                f"  [config] WARNING: unknown key '{key}' in config file "
                f"— ignored (check spelling or section name)."
            )

    return DQNConfig(**kwargs)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/training_config.json") -> DQNConfig:
    """Load, validate, and return a ``DQNConfig`` from an external JSON file.

    This is the single entry point for the external config system.
    Call it at the start of ``train.py`` instead of constructing
    ``DQNConfig()`` manually.

    Processing steps
    -----------------
    1. Resolve the config path (supports relative paths from project root).
    2. Read and JSON-parse the file.
    3. Flatten nested sections into a single dict.
    4. Validate every value.
    5. Merge with ``DQNConfig`` defaults.
    6. Return the final ``DQNConfig`` instance.

    Args:
        config_path (str): Path to the JSON config file.
                           Relative paths are resolved from the current
                           working directory.
                           Default: ``"config/training_config.json"``.

    Returns:
        DQNConfig: Fully populated, validated configuration instance.

    Raises:
        ConfigFileError:       If the file does not exist or is not valid JSON.
        ConfigValidationError: If any hyperparameter value fails its constraint.
    """
    # ── Resolve path ─────────────────────────────────────────────────────
    abs_path = os.path.abspath(config_path)

    if not os.path.isfile(abs_path):
        raise ConfigFileError(
            abs_path,
            "file not found. Run from the project root or pass an absolute path."
        )

    # ── Read and parse JSON ───────────────────────────────────────────────
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigFileError(
            abs_path,
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigFileError(abs_path, "top-level JSON must be an object ({})")

    print(f"  [config] Loaded external config → {abs_path}")

    # ── Flatten nested sections ───────────────────────────────────────────
    flat = flatten_config(raw)

    if not flat:
        print("  [config] WARNING: config file is empty — using all defaults.")
        return DQNConfig()

    # ── Validate ──────────────────────────────────────────────────────────
    validate_config(flat)

    # ── Merge with defaults and return ───────────────────────────────────
    cfg = merge_with_defaults(flat)

    print(
        f"  [config] Applied {len(flat)} override(s): "
        f"lr={cfg.learning_rate}  γ={cfg.gamma}  ε={cfg.epsilon_start}→{cfg.epsilon_end}"
        f"  episodes={cfg.max_episodes}"
    )

    return cfg


# ---------------------------------------------------------------------------
# CLI — print the resolved config without running training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import dataclasses

    path = sys.argv[1] if len(sys.argv) > 1 else "config/training_config.json"

    try:
        cfg = load_config(path)
    except (ConfigFileError, ConfigValidationError) as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    print("\n  Resolved DQNConfig:")
    print("  " + "-" * 50)
    for f in dataclasses.fields(cfg):
        print(f"  {f.name:<28}: {getattr(cfg, f.name)}")
    print("  " + "-" * 50)
