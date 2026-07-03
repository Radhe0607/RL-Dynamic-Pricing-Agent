"""
src/inference/__init__.py
-------------------------
Inference package for the DQN Dynamic Pricing Agent.

Exposes the public API for loading a trained checkpoint and
producing price recommendations from live state observations.

Public API
----------
    PricingPredictor              — main class for model inference
    PricingPredictor.from_checkpoint(path, cfg) → PricingPredictor
    predictor.predict(state)      → InferenceResult
    predictor.predict_batch(states) → list[InferenceResult]

Usage
-----
    from src.inference import PricingPredictor

    predictor = PricingPredictor.from_checkpoint("checkpoints/dqn_final.pt")
    result    = predictor.predict([50.0, 80.0, 30.0])
    print(result.recommended_price)   # e.g. 55.0
"""

from src.inference.predictor import PricingPredictor, InferenceResult

__all__ = ["PricingPredictor", "InferenceResult"]
