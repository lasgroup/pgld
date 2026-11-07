"""Tests for reward functions."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards import AllModelsHitReward


class TestAllModelsHitRewardFractionalCredit:
    """Tests for fractional credit assignment in AllModelsHitReward."""

    @pytest.fixture
    def mock_landscape(self):
        """Mock landscape with single objective."""

        class MockLandscape:
            active_objs = ["obj1"]

        return MockLandscape()

    def test_credit_split_among_all_templates_that_have_hit(self, mock_landscape):
        """
        If hit X exists in templates A and B, credit should be 1/2 each,
        even if only one template sampled it.
        """
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        batch_info = {
            "keys_scored": [{"hit_X"}, set()],  # A sampled, B didn't
            "key_counts": [{"hit_X": 5}, {"hit_X": 3}],  # Both HAVE hit_X
            "importance_weight": [1.0, 1.0],
            "batch_size": 100,
            "unit_sizes": [50, 50],
            "key_to_fitness": {"hit_X": 0.9},  # Above threshold
            "landscape": mock_landscape,
            "reward_unit": torch.tensor([0, 0, 0, 1, 1]),  # Dummy
        }

        reward_fn.compute_with_cache(batch_info)

        # Template A should get 0.5 (not 1.0), because B also has the hit
        # Template B gets 0 because it didn't sample, but the denominator is still 2
        batch_info["reward_unit"] = torch.tensor([0])
        reward_a = reward_fn.compute_with_cache(batch_info)

        batch_info["reward_unit"] = torch.tensor([1])
        reward_b = reward_fn.compute_with_cache(batch_info)

        # A gets 0.5 * weight / batch_size = 0.5 * 1.0 / 100 = 0.005
        assert abs(reward_a - 0.005) < 1e-6
        # B gets 0 because it didn't sample the hit
        assert reward_b == 0.0

    def test_both_templates_sample_hit(self, mock_landscape):
        """If both templates sample the hit, each gets 0.5 credit."""
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        batch_info = {
            "keys_scored": [{"hit_X"}, {"hit_X"}],  # Both sampled
            "key_counts": [{"hit_X": 5}, {"hit_X": 3}],
            "importance_weight": [1.0, 1.0],
            "batch_size": 100,
            "unit_sizes": [50, 50],
            "key_to_fitness": {"hit_X": 0.9},
            "landscape": mock_landscape,
            "reward_unit": torch.tensor([0]),
        }

        reward_a = reward_fn.compute_with_cache(batch_info)

        batch_info["reward_unit"] = torch.tensor([1])
        reward_b = reward_fn.compute_with_cache(batch_info)

        # Each gets 0.5 * 1.0 / 100 = 0.005
        assert abs(reward_a - 0.005) < 1e-6
        assert abs(reward_b - 0.005) < 1e-6

    def test_hit_in_single_template(self, mock_landscape):
        """If only one template has a hit, it gets full credit."""
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        batch_info = {
            "keys_scored": [{"hit_X"}, set()],
            "key_counts": [{"hit_X": 5}, {}],  # Only A has hit_X
            "importance_weight": [1.0, 1.0],
            "batch_size": 100,
            "unit_sizes": [50, 50],
            "key_to_fitness": {"hit_X": 0.9},
            "landscape": mock_landscape,
            "reward_unit": torch.tensor([0]),
        }

        reward_a = reward_fn.compute_with_cache(batch_info)

        # A gets full credit: 1.0 * 1.0 / 100 = 0.01
        assert abs(reward_a - 0.01) < 1e-6

    def test_importance_weight_applied(self, mock_landscape):
        """Importance weight should scale the reward."""
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        batch_info = {
            "keys_scored": [{"hit_X"}],
            "key_counts": [{"hit_X": 5}],
            "importance_weight": [2.0],  # IS weight of 2
            "batch_size": 100,
            "unit_sizes": [100],
            "key_to_fitness": {"hit_X": 0.9},
            "landscape": mock_landscape,
            "reward_unit": torch.tensor([0]),
        }

        reward = reward_fn.compute_with_cache(batch_info)

        # 1.0 * 2.0 / 100 = 0.02
        assert abs(reward - 0.02) < 1e-6

    def test_unbiased_estimator_expectation(self, mock_landscape):
        """
        Verify the fix gives unbiased estimates.

        Scenario: Both A and B have hit X, each samples 50% independently.
        True reward = 1 hit / batch_size = 0.5. Expected estimate should match.
        """
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        # Simulate all 4 scenarios with equal probability (0.25 each)
        total_credits = []
        for a_sampled, b_sampled in [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ]:
            keys_scored_a = {"hit_X"} if a_sampled else set()
            keys_scored_b = {"hit_X"} if b_sampled else set()

            batch_info = {
                "keys_scored": [keys_scored_a, keys_scored_b],
                "key_counts": [{"hit_X": 1}, {"hit_X": 1}],
                "importance_weight": [2.0, 2.0],  # 50% sampling -> weight 2
                "batch_size": 2,
                "unit_sizes": [1, 1],
                "key_to_fitness": {"hit_X": 0.9},
                "landscape": mock_landscape,
                "reward_unit": torch.tensor([0]),
            }

            batch_info["reward_unit"] = torch.tensor([0])
            credit_a = reward_fn.compute_with_cache(batch_info).item()

            batch_info["reward_unit"] = torch.tensor([1])
            credit_b = reward_fn.compute_with_cache(batch_info).item()

            total_credits.append(credit_a + credit_b)

        # Scenarios: [0.5, 0.5, 1.0, 0.0] -> mean = 0.5
        # This equals true reward: 1 hit / batch_size(2) = 0.5
        expected_total = sum(total_credits) / 4
        assert abs(expected_total - 0.5) < 1e-6, f"Expected 0.5, got {expected_total}"

    def test_multiple_hits_different_templates(self, mock_landscape):
        """Multiple hits, some shared, some unique."""
        reward_fn = AllModelsHitReward(thresholds={"obj1": 0.5})

        batch_info = {
            # hit_X: both have, both sampled
            # hit_Y: only A has, A sampled
            # hit_Z: both have, only B sampled
            "keys_scored": [{"hit_X", "hit_Y"}, {"hit_X", "hit_Z"}],
            "key_counts": [
                {"hit_X": 2, "hit_Y": 3, "hit_Z": 2},  # A has hit_Z but didn't sample
                {"hit_X": 4, "hit_Z": 1},
            ],
            "importance_weight": [1.0, 1.0],
            "batch_size": 100,
            "unit_sizes": [50, 50],
            "key_to_fitness": {"hit_X": 0.9, "hit_Y": 0.8, "hit_Z": 0.7},
            "landscape": mock_landscape,
            "reward_unit": torch.tensor([0]),
        }

        reward_a = reward_fn.compute_with_cache(batch_info)

        batch_info["reward_unit"] = torch.tensor([1])
        reward_b = reward_fn.compute_with_cache(batch_info)

        # A: hit_X (0.5) + hit_Y (1.0) + hit_Z (0, didn't sample) = 1.5 -> 1.5 / 100 = 0.015
        # B: hit_X (0.5) + hit_Z (0.5) = 1.0 -> 1.0 / 100 = 0.01
        assert abs(reward_a - 0.015) < 1e-6
        assert abs(reward_b - 0.01) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
