#!/usr/bin/env python3
"""
Interactive visualization script for the multi-robot search problem
Shows searchers and target beliefs in real-time during planning
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from main import Mespp


class SearchVisualizer:
    """Interactive visualizer for multi-robot search"""
    
    def __init__(self):
        """Initialize the MILP problem and solver"""
        print("Initializing search problem...")
        self.mespp = Mespp()
        self.mespp.addMILPVariables()
        self.mespp.m.update()
        self.mespp.addMILPConstraints()
        self.mespp.setMILPObjective()
        
        print("Solving MILP...")
        self.mespp.plan()
        
        self.SIDE = 10  # Adjust if you changed SIDE in main.py
        self.current_time = 0
        
    def get_grid_position(self, vertex_id):
        """Convert vertex ID to (row, col) grid position"""
        return divmod(vertex_id, self.SIDE)
    
    def create_figure(self):
        """Create the main figure with subplots"""
        self.fig = plt.figure(figsize=(16, 6))
        self.fig.suptitle('Multi-Robot Efficient Search Path Planning', 
                         fontsize=16, fontweight='bold')
        
        # Left subplot: Position graph with searchers and target
        self.ax_pos = self.fig.add_subplot(131)
        self.ax_pos.set_title('Searcher Positions & Capture Coverage')
        
        # Middle subplot: Belief graph
        self.ax_belief = self.fig.add_subplot(132)
        self.ax_belief.set_title('Target Occupancy Belief')
        
        # Right subplot: Statistics
        self.ax_stats = self.fig.add_subplot(133)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics & Dynamics')
        
        for ax in [self.ax_pos, self.ax_belief]:
            ax.set_xlim(-0.5, self.SIDE - 0.5)
            ax.set_ylim(-0.5, self.SIDE - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, linestyle='--')
    
    def draw_grid_background(self, ax, color='lightgray'):
        """Draw grid background"""
        for i in range(self.SIDE + 1):
            ax.axhline(i - 0.5, color=color, linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color=color, linewidth=0.5, alpha=0.3)
    
    def update_frame(self, t):
        """Update visualization for time step t"""
        self.current_time = t
        
        # Clear axes
        self.ax_pos.clear()
        self.ax_belief.clear()
        self.ax_stats.clear()
        
        # Setup position subplot
        self.ax_pos.set_xlim(-0.5, self.SIDE - 0.5)
        self.ax_pos.set_ylim(-0.5, self.SIDE - 0.5)
        self.ax_pos.set_aspect('equal')
        self.ax_pos.invert_yaxis()
        self.ax_pos.set_title(f'Searcher Positions at t={t}', fontweight='bold')
        self.draw_grid_background(self.ax_pos)
        
        # Setup belief subplot
        self.ax_belief.set_xlim(-0.5, self.SIDE - 0.5)
        self.ax_belief.set_ylim(-0.5, self.SIDE - 0.5)
        self.ax_belief.set_aspect('equal')
        self.ax_belief.invert_yaxis()
        self.ax_belief.set_title(f'Target Belief Distribution at t={t}', fontweight='bold')
        self.draw_grid_background(self.ax_belief)
        
        # ===== LEFT SUBPLOT: Searchers and Capture Zones =====
        
        # Draw capture zones (where searchers can capture)
        for v in range(self.mespp.N):
            if v in self.mespp.legal_V[t]:
                for s in range(self.mespp.searchers.M):
                    if v in self.mespp.legal_V[t][s]:
                        capture_val = self.mespp.capture[v, t].X
                        if isinstance(capture_val, np.ndarray):
                            capture_val = capture_val.item() if capture_val.size == 1 else capture_val[0]
                        if capture_val > 0.5:
                            row, col = self.get_grid_position(v)
                            rect = patches.Rectangle((col - 0.45, row - 0.45), 0.9, 0.9,
                                                     linewidth=1, edgecolor='green', 
                                                     facecolor='lightgreen', alpha=0.3)
                            self.ax_pos.add_patch(rect)
        
        # Draw searchers
        for s, searcher_pos in enumerate(self.mespp.searchers.initial_positions):
            row, col = self.get_grid_position(searcher_pos)
            self.ax_pos.plot(col, row, 'g^', markersize=20, 
                           label=f'Searcher {s+1}', markeredgecolor='darkgreen', 
                           markeredgewidth=2)
        
        # Draw target
        target_row, target_col = self.get_grid_position(self.mespp.target.position)
        self.ax_pos.plot(target_col, target_row, 'r*', markersize=30,
                        label='Target (actual)', markeredgecolor='darkred', 
                        markeredgewidth=1)
        
        self.ax_pos.legend(loc='upper right', fontsize=9)
        self.ax_pos.set_xticks(range(self.SIDE))
        self.ax_pos.set_yticks(range(self.SIDE))
        
        # ===== MIDDLE SUBPLOT: Belief Heat Map =====
        
        # Extract belief array and reshape to grid
        belief_list = []
        for v in range(self.mespp.N):
            b_val = self.mespp.beliefs[v+1, t].X
            if isinstance(b_val, np.ndarray):
                b_val = b_val.item() if b_val.size == 1 else b_val[0]
            belief_list.append(b_val)
        belief_array = np.array(belief_list)
        belief_grid = belief_array.reshape(self.SIDE, self.SIDE)
        
        # Create heat map
        max_belief = np.max(belief_grid) if np.max(belief_grid) > 0 else 1
        im = self.ax_belief.imshow(belief_grid, cmap='hot', vmin=0, vmax=max_belief,
                                   aspect='auto', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax_belief, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=15)
        
        self.ax_belief.set_xticks(range(self.SIDE))
        self.ax_belief.set_yticks(range(self.SIDE))
        
        # ===== RIGHT SUBPLOT: Statistics =====
        
        self.ax_stats.axis('off')
        
        # Calculate statistics
        capture_belief_val = self.mespp.beliefs[0, t].X
        if isinstance(capture_belief_val, np.ndarray):
            capture_belief = capture_belief_val.item() if capture_belief_val.size == 1 else capture_belief_val[0]
        else:
            capture_belief = capture_belief_val
        max_occupancy_belief = np.max(belief_grid)
        num_searchers = self.mespp.searchers.M
        time_step = t
        total_horizon = self.mespp.HORIZON
        
        # Format statistics text
        stats_text = f"""
PROBLEM PARAMETERS
━━━━━━━━━━━━━━━━━━━
Grid Size: {self.SIDE}×{self.SIDE} ({self.mespp.N} vertices)
Number of Searchers: {num_searchers}
Time Horizon: {total_horizon}

CURRENT STATE (t={time_step})
━━━━━━━━━━━━━━━━━━━
Target Position: {self.mespp.target.position}
Target Grid Coords: ({target_row}, {target_col})

BELIEFS & CAPTURE
━━━━━━━━━━━━━━━━━━━
Cumulative Capture Belief: {capture_belief:.4f}
Max Occupancy Belief: {max_occupancy_belief:.4f}
Num Cells with Belief > 0: {np.sum(belief_grid > 0.01)}

SEARCH COVERAGE
━━━━━━━━━━━━━━━━━━━
Reachable Vertices (t={time_step}):
  Searcher coverage radius: {time_step}
  
        """
        
        self.ax_stats.text(0.05, 0.95, stats_text, 
                          transform=self.ax_stats.transAxes,
                          fontfamily='monospace',
                          fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Update target position for next frame
        if t < self.mespp.HORIZON:
            self.mespp.target.updateTargetPosition()
        
        return [self.ax_pos, self.ax_belief, self.ax_stats]
    
    def animate(self):
        """Run the animation"""
        print(f"\nStarting animation for {self.mespp.HORIZON + 1} time steps...")
        
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_frame,
            frames=self.mespp.HORIZON + 1,
            interval=800,  # 800ms per frame
            repeat=True,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  Multi-Robot Search - Interactive Visualization")
    print("="*60 + "\n")
    
    try:
        visualizer = SearchVisualizer()
        visualizer.create_figure()
        visualizer.animate()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
