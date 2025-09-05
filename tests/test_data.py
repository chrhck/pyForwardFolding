"""
Expected values for statistical tests.

This module contains all expected numerical values used in statistical tests.
These values were captured from the actual implementation and serve as
regression test baselines. Update these values only when intentional
changes are made to the underlying algorithms.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpectedResults:
    """Container for all expected test results."""
    
    # Hypothesis evaluation results
    hypothesis_eval_basic: float = 13.318561553955078
    hypothesis_eval_params_1_5_0_8: float = 17.999671936035156
    hypothesis_eval_params_2_0_1_2: float = 26.724206924438477
    
    # Best fit parameters for detailed evaluation
    best_fit_param1: float = 1.0221792185966851
    best_fit_param2: float = 1.0398706843467973
    best_fit_param3: float = 1.0  # Fixed value
    
    # Test statistics
    test_stat: float = 3.9552879333496094
    test_stat_with_param1_1_2: float = 0.0
    test_stat_different_hypotheses: float = 0.0
    
    # Scan results
    scan_grid_5_points: List[float] = field(default_factory=lambda: [0.1, 2.575, 5.05, 7.525, 10.0])
    scan_ts_values_5_points: List[float] = field(default_factory= lambda: [74.79598 ,  29.627384, 118.353386, 221.36356 , 330.32715])
    
   
    # Power calculation bounds
    power_lower_bound: float = 0.3
    power_upper_bound: float = 0.8


@dataclass
class MockDistributions:
    """Mock distribution values for testing."""
    
    null_dist_5_points: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5])
    alt_dist_5_points: List[float] = field(default_factory=lambda: [1.8, 2.2, 2.5, 3.0, 1.0])


# Global instances
EXPECTED = ExpectedResults()
MOCK_DISTS = MockDistributions()
