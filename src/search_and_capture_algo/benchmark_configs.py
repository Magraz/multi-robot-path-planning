#!/usr/bin/env python3
"""
Utility script to run multiple configurations and compare results
Useful for benchmarking different grid sizes and search scenarios
"""

import os
import sys
import time
import numpy as np
import random

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from main import Mespp


class ConfigurationRunner:
    """Run multiple search configurations and collect statistics"""
    
    def __init__(self):
        self.results = []
    
    def run_configuration(self, name, side, horizon, num_searchers, 
                         searcher_positions, target_pos, target_motion="uniform",
                         circular_grid=False):
        """
        Run a single configuration
        
        Args:
            name: Configuration name
            side: Grid side length (SIDE×SIDE)
            horizon: Time horizon
            num_searchers: Number of searchers
            searcher_positions: Array of initial searcher positions
            target_pos: Initial target position
            target_motion: "uniform" or "stationary"
            circular_grid: Whether grid wraps around (toroidal)
        
        Returns:
            dict: Configuration result statistics
        """
        
        print(f"\n{'='*70}")
        print(f"Configuration: {name}")
        print(f"{'='*70}")
        print(f"  Grid: {side}×{side} ({side*side} vertices)")
        print(f"  Topology: {'Toroidal' if circular_grid else 'Planar'}")
        print(f"  Searchers: {num_searchers}")
        print(f"  Initial positions: {searcher_positions}")
        print(f"  Target: pos={target_pos}, motion={target_motion}")
        print(f"  Horizon: {horizon}")
        
        # Set up problem
        mespp = Mespp()
        mespp.SIDE = side
        mespp.HORIZON = horizon
        mespp.N = side * side
        
        import igraph as ig
        mespp.g = ig.Graph.Lattice(dim=[side, side], circular=circular_grid)
        mespp.belief_g = mespp.g.copy()
        
        # Create target
        from target import Target
        mespp.target = Target(mespp.g, N=mespp.N, 
                             initial_position=target_pos, 
                             motion=target_motion)
        
        # Create searchers
        from searchers import Searchers
        mespp.searchers = Searchers(mespp.g, N=mespp.N, M=num_searchers,
                                   initial_positions=searcher_positions,
                                   target_initial_position=target_pos)
        
        # Add variables
        print("\n  Adding MILP variables...")
        mespp.addMILPVariables()
        mespp.m.update()
        
        # Add constraints
        print("  Adding MILP constraints...")
        mespp.addMILPConstraints()
        mespp.setMILPObjective()
        
        # Solve
        print("  Solving MILP...")
        start_time = time.time()
        mespp.plan()
        solve_time = time.time() - start_time
        
        # Extract results
        try:
            optimal_value = mespp.m.objVal if mespp.m.status == 2 else None  # 2 = OPTIMAL
            capture_belief = mespp.beliefs[0, horizon].X[0]
        except:
            optimal_value = None
            capture_belief = None
        
        result = {
            'name': name,
            'grid_size': side,
            'num_vertices': side * side,
            'topology': 'toroidal' if circular_grid else 'planar',
            'num_searchers': num_searchers,
            'horizon': horizon,
            'target_motion': target_motion,
            'solve_time': solve_time,
            'objective_value': optimal_value,
            'capture_belief_final': capture_belief,
            'status': 'OPTIMAL' if mespp.m.status == 2 else 'SUBOPTIMAL/FAILED'
        }
        
        print(f"\n  Results:")
        print(f"    Status: {result['status']}")
        print(f"    Solve time: {solve_time:.2f}s")
        print(f"    Objective value: {optimal_value}")
        print(f"    Final capture belief: {capture_belief}")
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print summary table of all results"""
        
        print(f"\n\n{'='*100}")
        print("CONFIGURATION COMPARISON SUMMARY")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Config':<20} {'Grid':<8} {'Searchers':<10} {'Motion':<12} "
              f"{'Time(s)':<10} {'Objective':<12} {'Status':<10}")
        print("-" * 100)
        
        # Rows
        for r in self.results:
            print(f"{r['name']:<20} "
                  f"{r['grid_size']}x{r['grid_size']:<5} "
                  f"{r['num_searchers']:<10} "
                  f"{r['target_motion']:<12} "
                  f"{r['solve_time']:<10.2f} "
                  f"{r['objective_value']:<12.4f}" if r['objective_value'] else f"{'N/A':<12} "
                  f"{r['status']:<10}")
        
        print(f"\n{'='*100}\n")


def main():
    """Run benchmark suite"""
    
    print("\n" + "="*70)
    print("Multi-Robot Search - Configuration Benchmark Suite")
    print("="*70 + "\n")
    
    runner = ConfigurationRunner()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Configuration 1: Small and fast
        runner.run_configuration(
            name="Small Grid (5x5)",
            side=5,
            horizon=8,
            num_searchers=1,
            searcher_positions=np.array([0]),
            target_pos=12,
            target_motion="uniform"
        )
        
        # Configuration 2: Default
        runner.run_configuration(
            name="Default (10x10)",
            side=10,
            horizon=10,
            num_searchers=2,
            searcher_positions=np.array([0, 99]),
            target_pos=50,
            target_motion="uniform"
        )
        
        # Configuration 3: Stationary target
        runner.run_configuration(
            name="Stationary Target",
            side=10,
            horizon=10,
            num_searchers=2,
            searcher_positions=np.array([0, 99]),
            target_pos=50,
            target_motion="stationary"
        )
        
        # Configuration 4: More searchers
        runner.run_configuration(
            name="Three Searchers",
            side=10,
            horizon=10,
            num_searchers=3,
            searcher_positions=np.array([0, 9, 50]),
            target_pos=50,
            target_motion="uniform"
        )
        
        # Configuration 5: Larger grid
        runner.run_configuration(
            name="Large Grid (12x12)",
            side=12,
            horizon=10,
            num_searchers=2,
            searcher_positions=np.array([0, 143]),
            target_pos=72,
            target_motion="uniform"
        )
        
        # Configuration 6: Toroidal grid
        runner.run_configuration(
            name="Toroidal Grid",
            side=10,
            horizon=10,
            num_searchers=2,
            searcher_positions=np.array([0, 99]),
            target_pos=50,
            target_motion="uniform",
            circular_grid=True
        )
        
        # Print summary
        runner.print_summary()
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
