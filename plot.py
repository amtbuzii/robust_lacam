"""Visualization function for MAPF solutions."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.colors import ListedColormap

from src.pycam.mapf_utils import Config, Configs, Grid


def plot_solution(grid: Grid, solution: Configs, title: str = "MAPF Solution") -> None:
    """
    Plot a MAPF solution on a grid.
    
    Args:
        grid: Grid array where True = free cell, False = obstacle
        solution: List of Config objects representing the solution path
        title: Title for the plot
    """
    if not solution or len(solution) == 0:
        print("Warning: Empty solution, nothing to plot")
        return
    
    height, width = grid.shape
    num_agents = len(solution[0]) if solution else 0
    
    if num_agents == 0:
        print("Warning: No agents in solution")
        return
    
    # Create figure and axis - limit size for very large grids
    max_fig_size = 20
    fig_width = min(max(8, width * 0.05), max_fig_size)
    fig_height = min(max(8, height * 0.05), max_fig_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create grid visualization: 0 = obstacle (gray), 1 = free (white)
    grid_vis = np.zeros_like(grid, dtype=float)
    grid_vis[grid] = 1.0  # Free cells are white (1.0)
    grid_vis[~grid] = 0.0  # Obstacles are gray (0.0)
    
    # Display the grid with custom colormap
    cmap = ListedColormap(['gray', 'white'])
    ax.imshow(grid_vis, cmap=cmap, vmin=0, vmax=1, origin='upper')
    
    # Draw grid lines in black - only for smaller grids (otherwise too dense/slow)
    max_grid_lines = 50  # Only draw grid lines if grid dimensions are reasonable
    if width <= max_grid_lines and height <= max_grid_lines:
        for y in range(height + 1):
            ax.axhline(y - 0.5, color='black', linewidth=0.5)
        for x in range(width + 1):
            ax.axvline(x - 0.5, color='black', linewidth=0.5)
    
    # Define colors for each agent (use distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_agents)))
    if num_agents > 10:
        # If more than 10 agents, use additional colors
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_agents)))
    
    # Extract paths for each agent
    agent_paths = [[] for _ in range(num_agents)]
    for config in solution:
        for agent_id in range(num_agents):
            y, x = config[agent_id]
            agent_paths[agent_id].append((x, y))  # Note: matplotlib uses (x, y)
    
    # Draw paths for each agent
    for agent_id in range(num_agents):
        path = agent_paths[agent_id]
        if len(path) > 1:
            # Draw the path as a line
            xs, ys = zip(*path)
            ax.plot(xs, ys, color=colors[agent_id % len(colors)], 
                   linewidth=2, alpha=0.7, label=f'Agent {agent_id + 1}')
            # Draw start position with a square marker
            if path:
                ax.scatter([xs[0]], [ys[0]], color=colors[agent_id % len(colors)], 
                          marker='s', s=200, edgecolors='black', linewidths=1.5, 
                          zorder=5, label=f'Start {agent_id + 1}' if agent_id == 0 else '')
                # Draw end position with a star marker
                ax.scatter([xs[-1]], [ys[-1]], color=colors[agent_id % len(colors)], 
                          marker='*', s=300, edgecolors='black', linewidths=1.5, 
                          zorder=5, label=f'Goal {agent_id + 1}' if agent_id == 0 else '')
        elif len(path) == 1:
            # Single point (no movement)
            x, y = path[0]
            ax.scatter([x], [y], color=colors[agent_id % len(colors)], 
                      marker='o', s=200, edgecolors='black', linewidths=1.5, 
                      zorder=5, label=f'Agent {agent_id + 1}')
    
    # Set axis properties
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # Invert y-axis to match grid coordinates
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    
    # Add grid labels (optional, can be commented out for large grids)
    if width <= 20 and height <= 20:
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_xticklabels(range(width))
        ax.set_yticklabels(range(height))
    else:
        # For large grids, show fewer ticks
        ax.set_xticks(range(0, width, max(1, width // 10)))
        ax.set_yticks(range(0, height, max(1, height // 10)))
    
    # Add legend if not too many agents
    if num_agents <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    plt.tight_layout()
    plt.show()


def animate_solution(
    grid: Grid,
    solution: Configs,
    title: str = "MAPF Solution Animation",
    interval: int = 500,
    save_gif: str | None = None,
) -> None:
    """
    Animate a MAPF solution showing robots moving on the grid over time.
    
    Args:
        grid: Grid array where True = free cell, False = obstacle
        solution: List of Config objects representing the solution path
        title: Title for the animation
        interval: Time between frames in milliseconds
        save_gif: Optional filename to save the animation as a GIF
    """
    if not solution or len(solution) == 0:
        print("Warning: Empty solution, nothing to animate")
        return
    
    height, width = grid.shape
    num_agents = len(solution[0]) if solution else 0
    
    if num_agents == 0:
        print("Warning: No agents in solution")
        return
    
    # Create figure and axis
    max_fig_size = 20
    fig_width = min(max(8, width * 0.05), max_fig_size)
    fig_height = min(max(8, height * 0.05), max_fig_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create grid visualization
    grid_vis = np.zeros_like(grid, dtype=float)
    grid_vis[grid] = 1.0
    grid_vis[~grid] = 0.0
    
    cmap = ListedColormap(['gray', 'white'])
    ax.imshow(grid_vis, cmap=cmap, vmin=0, vmax=1, origin='upper')
    
    # Draw grid lines for smaller grids
    max_grid_lines = 50
    if width <= max_grid_lines and height <= max_grid_lines:
        for y in range(height + 1):
            ax.axhline(y - 0.5, color='black', linewidth=0.5)
        for x in range(width + 1):
            ax.axvline(x - 0.5, color='black', linewidth=0.5)
    
    # Define colors for each agent
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_agents)))
    if num_agents > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_agents)))
    
    # Extract paths and start/goal positions
    agent_paths = [[] for _ in range(num_agents)]
    starts = []
    goals = []
    
    for config in solution:
        for agent_id in range(num_agents):
            y, x = config[agent_id]
            agent_paths[agent_id].append((x, y))
    
    # Get start and goal positions
    for agent_id in range(num_agents):
        if agent_paths[agent_id]:
            starts.append(agent_paths[agent_id][0])
            goals.append(agent_paths[agent_id][-1])
    
    # Draw start positions (squares)
    start_xs, start_ys = zip(*starts) if starts else ([], [])
    start_scatter = ax.scatter(
        start_xs, start_ys, 
        color=[colors[i % len(colors)] for i in range(num_agents)],
        marker='s', s=300, edgecolors='black', linewidths=2,
        zorder=5, label='Start'
    )
    
    # Draw goal positions (stars)
    goal_xs, goal_ys = zip(*goals) if goals else ([], [])
    goal_scatter = ax.scatter(
        goal_xs, goal_ys,
        color=[colors[i % len(colors)] for i in range(num_agents)],
        marker='*', s=400, edgecolors='black', linewidths=2,
        zorder=5, label='Goal'
    )
    
    # Initialize agent positions (circles)
    agent_scatters = []
    path_lines = []
    for agent_id in range(num_agents):
        # Agent position marker
        scatter = ax.scatter(
            [], [], 
            color=colors[agent_id % len(colors)],
            marker='o', s=250, edgecolors='black', linewidths=2,
            zorder=6, label=f'Agent {agent_id + 1}'
        )
        agent_scatters.append(scatter)
        
        # Path line (will be drawn progressively)
        line, = ax.plot([], [], 
                       color=colors[agent_id % len(colors)],
                       linewidth=2, alpha=0.5, zorder=4)
        path_lines.append(line)
    
    # Set axis properties
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate', fontsize=10)
    ax.set_ylabel('Y coordinate', fontsize=10)
    
    # Add grid labels for small grids
    if width <= 20 and height <= 20:
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_xticklabels(range(width))
        ax.set_yticklabels(range(height))
    
    # Time step text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate_frame(frame):
        """Update animation for each frame."""
        # Update time step text
        time_text.set_text(f'Time Step: {frame}/{len(solution) - 1}')
        
        # Update each agent's position and path
        for agent_id in range(num_agents):
            if frame < len(agent_paths[agent_id]):
                # Current position
                x, y = agent_paths[agent_id][frame]
                agent_scatters[agent_id].set_offsets([[x, y]])
                
                # Path so far (from start to current position)
                if frame > 0:
                    path_so_far = agent_paths[agent_id][:frame + 1]
                    xs, ys = zip(*path_so_far)
                    path_lines[agent_id].set_data(xs, ys)
                else:
                    path_lines[agent_id].set_data([], [])
            else:
                # Agent has reached goal, keep it at goal position
                if agent_paths[agent_id]:
                    x, y = agent_paths[agent_id][-1]
                    agent_scatters[agent_id].set_offsets([[x, y]])
        
        return agent_scatters + path_lines + [time_text]
    
    # Create animation
    num_frames = len(solution)
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=num_frames,
        interval=interval, blit=True, repeat=True
    )
    
    # Add legend if not too many agents
    if num_agents <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    plt.tight_layout()
    
    # Save as GIF if requested
    if save_gif:
        print(f"Saving animation to {save_gif}...")
        try:
            anim.save(save_gif, writer='pillow', fps=1000/interval)
            print(f"Animation saved to {save_gif}")
        except Exception as e:
            print(f"Warning: Could not save GIF: {e}")
            print("Animation will still play in the window.")
    
    plt.show()
    
    return anim


def plot_solutions_comparison(
    grid: Grid,
    solution1: Configs,
    solution2: Configs,
    title1: str = "Solution 1",
    title2: str = "Solution 2",
    main_title: str = "Solution Comparison",
) -> None:
    """
    Plot two MAPF solutions side by side for comparison.
    If only one solution exists, plot just that one.
    
    Args:
        grid: Grid array where True = free cell, False = obstacle
        solution1: First solution (list of Config objects)
        solution2: Second solution (list of Config objects)
        title1: Title for the first plot
        title2: Title for the second plot
        main_title: Main title for the entire figure
    """
    has_solution1 = solution1 and len(solution1) > 0
    has_solution2 = solution2 and len(solution2) > 0
    
    # If neither solution exists, return
    if not has_solution1 and not has_solution2:
        print("Warning: No solutions to plot")
        return
    
    # If only one solution exists, use the single plot function
    if has_solution1 and not has_solution2:
        print("Note: Only first solution available, plotting single solution...")
        plot_solution(grid, solution1, title=title1)
        return
    
    if has_solution2 and not has_solution1:
        print("Note: Only second solution available, plotting single solution...")
        plot_solution(grid, solution2, title=title2)
        return
    
    height, width = grid.shape
    num_agents1 = len(solution1[0]) if solution1 else 0
    num_agents2 = len(solution2[0]) if solution2 else 0
    
    if num_agents1 == 0 or num_agents2 == 0:
        print("Warning: No agents in solution(s)")
        return
    
    if num_agents1 != num_agents2:
        print(f"Warning: Different number of agents ({num_agents1} vs {num_agents2})")
    
    num_agents = max(num_agents1, num_agents2)
    
    # Create figure with two subplots side by side - limit size for very large grids
    max_fig_size = 20
    fig_width = min(max(16, width * 0.1), max_fig_size * 2)
    fig_height = min(max(8, height * 0.1), max_fig_size)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Create grid visualization: 0 = obstacle (gray), 1 = free (white)
    grid_vis = np.zeros_like(grid, dtype=float)
    grid_vis[grid] = 1.0  # Free cells are white (1.0)
    grid_vis[~grid] = 0.0  # Obstacles are gray (0.0)
    
    # Custom colormap
    cmap = ListedColormap(['gray', 'white'])
    
    # Define colors for each agent (use distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_agents)))
    if num_agents > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_agents)))
    
    # Helper function to plot a solution on an axis
    def plot_solution_on_axis(ax, solution, title, num_agents_sol):
        # Display the grid
        ax.imshow(grid_vis, cmap=cmap, vmin=0, vmax=1, origin='upper')
        
        # Draw grid lines in black - only for smaller grids (otherwise too dense/slow)
        max_grid_lines = 50  # Only draw grid lines if grid dimensions are reasonable
        if width <= max_grid_lines and height <= max_grid_lines:
            for y in range(height + 1):
                ax.axhline(y - 0.5, color='black', linewidth=0.5)
            for x in range(width + 1):
                ax.axvline(x - 0.5, color='black', linewidth=0.5)
        
        # Extract paths for each agent
        agent_paths = [[] for _ in range(num_agents_sol)]
        for config in solution:
            for agent_id in range(num_agents_sol):
                y, x = config[agent_id]
                agent_paths[agent_id].append((x, y))  # Note: matplotlib uses (x, y)
        
        # Draw paths for each agent
        for agent_id in range(num_agents_sol):
            path = agent_paths[agent_id]
            if len(path) > 1:
                # Draw the path as a line
                xs, ys = zip(*path)
                ax.plot(xs, ys, color=colors[agent_id % len(colors)], 
                       linewidth=2, alpha=0.7, label=f'Agent {agent_id + 1}')
                # Draw start position with a square marker
                if path:
                    ax.scatter([xs[0]], [ys[0]], color=colors[agent_id % len(colors)], 
                              marker='s', s=200, edgecolors='black', linewidths=1.5, 
                              zorder=5)
                    # Draw end position with a star marker
                    ax.scatter([xs[-1]], [ys[-1]], color=colors[agent_id % len(colors)], 
                              marker='*', s=300, edgecolors='black', linewidths=1.5, 
                              zorder=5)
            elif len(path) == 1:
                # Single point (no movement)
                x, y = path[0]
                ax.scatter([x], [y], color=colors[agent_id % len(colors)], 
                          marker='o', s=200, edgecolors='black', linewidths=1.5, 
                          zorder=5, label=f'Agent {agent_id + 1}')
        
        # Set axis properties
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)  # Invert y-axis to match grid coordinates
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X coordinate', fontsize=9)
        ax.set_ylabel('Y coordinate', fontsize=9)
        
        # Add grid labels (optional, for small grids)
        if width <= 20 and height <= 20:
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
            ax.set_xticklabels(range(width))
            ax.set_yticklabels(range(height))
        else:
            # For large grids, show fewer ticks
            ax.set_xticks(range(0, width, max(1, width // 10)))
            ax.set_yticks(range(0, height, max(1, height // 10)))
        
        # Add legend if not too many agents
        if num_agents_sol <= 10:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    
    # Plot both solutions
    plot_solution_on_axis(ax1, solution1, title1, num_agents1)
    plot_solution_on_axis(ax2, solution2, title2, num_agents2)
    
    # Add cost information as text
    cost1 = len(solution1) - 1 if solution1 else 0
    cost2 = len(solution2) - 1 if solution2 else 0
    ax1.text(0.02, 0.98, f'Cost: {cost1}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.02, 0.98, f'Cost: {cost2}', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_three_solutions_comparison(
    grid: Grid,
    solution1: Configs,
    solution2: Configs,
    solution3: Configs,
    title1: str = "Solution 1",
    title2: str = "Solution 2",
    title3: str = "Solution 3",
    main_title: str = "Three Solutions Comparison",
    soc1: int | None = None,
    soc2: int | None = None,
    soc3: int | None = None,
    runtime1: float | None = None,
    runtime2: float | None = None,
    runtime3: float | None = None,
    solution_time1: float | None = None,
    solution_time2: float | None = None,
    solution_time3: float | None = None,
) -> None:
    """
    Plot three MAPF solutions side by side for comparison.
    
    Args:
        grid: Grid array where True = free cell, False = obstacle
        solution1: First solution (list of Config objects)
        solution2: Second solution (list of Config objects)
        solution3: Third solution (list of Config objects)
        title1: Title for the first plot
        title2: Title for the second plot
        title3: Title for the third plot
        main_title: Main title for the entire figure
        soc1: Sum of Costs for solution1
        soc2: Sum of Costs for solution2
        soc3: Sum of Costs for solution3
        runtime1: Runtime in milliseconds for solution1
        runtime2: Runtime in milliseconds for solution2
        runtime3: Runtime in milliseconds for solution3
    """
    has_solution1 = solution1 and len(solution1) > 0
    has_solution2 = solution2 and len(solution2) > 0
    has_solution3 = solution3 and len(solution3) > 0
    
    # If no solutions exist, return
    if not (has_solution1 or has_solution2 or has_solution3):
        print("Warning: No solutions to plot")
        return
    
    height, width = grid.shape
    num_agents1 = len(solution1[0]) if solution1 and len(solution1) > 0 else 0
    num_agents2 = len(solution2[0]) if solution2 and len(solution2) > 0 else 0
    num_agents3 = len(solution3[0]) if solution3 and len(solution3) > 0 else 0
    
    if num_agents1 == 0 and num_agents2 == 0 and num_agents3 == 0:
        print("Warning: No agents in solutions")
        return
    
    num_agents = max(num_agents1, num_agents2, num_agents3)
    
    # Create figure with three subplots side by side - limit size for very large grids
    max_fig_size = 20
    fig_width = min(max(24, width * 0.15), max_fig_size * 3)
    fig_height = min(max(8, height * 0.1), max_fig_size)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Create grid visualization: 0 = obstacle (gray), 1 = free (white)
    grid_vis = np.zeros_like(grid, dtype=float)
    grid_vis[grid] = 1.0  # Free cells are white (1.0)
    grid_vis[~grid] = 0.0  # Obstacles are gray (0.0)
    
    # Custom colormap
    cmap = ListedColormap(['gray', 'white'])
    
    # Define colors for each agent (use distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_agents)))
    if num_agents > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_agents)))
    
    # Helper function to plot a solution on an axis
    def plot_solution_on_axis(ax, solution, title, num_agents_sol):
        if not solution or num_agents_sol == 0:
            ax.text(0.5, 0.5, 'No solution', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        
        # Display the grid
        ax.imshow(grid_vis, cmap=cmap, vmin=0, vmax=1, origin='upper')
        
        # Draw grid lines in black - only for smaller grids (otherwise too dense/slow)
        max_grid_lines = 50  # Only draw grid lines if grid dimensions are reasonable
        if width <= max_grid_lines and height <= max_grid_lines:
            for y in range(height + 1):
                ax.axhline(y - 0.5, color='black', linewidth=0.5)
            for x in range(width + 1):
                ax.axvline(x - 0.5, color='black', linewidth=0.5)
        
        # Extract paths for each agent
        agent_paths = [[] for _ in range(num_agents_sol)]
        for config in solution:
            for agent_id in range(num_agents_sol):
                y, x = config[agent_id]
                agent_paths[agent_id].append((x, y))  # Note: matplotlib uses (x, y)
        
        # Draw paths for each agent
        for agent_id in range(num_agents_sol):
            path = agent_paths[agent_id]
            if len(path) > 1:
                # Draw the path as a line
                xs, ys = zip(*path)
                ax.plot(xs, ys, color=colors[agent_id % len(colors)], 
                       linewidth=2, alpha=0.7, label=f'Agent {agent_id + 1}')
                # Draw start position with a square marker
                if path:
                    ax.scatter([xs[0]], [ys[0]], color=colors[agent_id % len(colors)], 
                              marker='s', s=200, edgecolors='black', linewidths=1.5, 
                              zorder=5)
                    # Draw end position with a star marker
                    ax.scatter([xs[-1]], [ys[-1]], color=colors[agent_id % len(colors)], 
                              marker='*', s=300, edgecolors='black', linewidths=1.5, 
                              zorder=5)
            elif len(path) == 1:
                # Single point (no movement)
                x, y = path[0]
                ax.scatter([x], [y], color=colors[agent_id % len(colors)], 
                          marker='o', s=200, edgecolors='black', linewidths=1.5, 
                          zorder=5, label=f'Agent {agent_id + 1}')
        
        # Set axis properties
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)  # Invert y-axis to match grid coordinates
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X coordinate', fontsize=9)
        ax.set_ylabel('Y coordinate', fontsize=9)
        
        # Add grid labels (optional, for small grids)
        if width <= 20 and height <= 20:
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
            ax.set_xticklabels(range(width))
            ax.set_yticklabels(range(height))
        else:
            # For large grids, show fewer ticks
            ax.set_xticks(range(0, width, max(1, width // 10)))
            ax.set_yticks(range(0, height, max(1, height // 10)))
        
        # Add legend if not too many agents
        if num_agents_sol <= 10:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    
    # Extract paths for animations
    def extract_paths(solution, num_agents_sol):
        if not solution or num_agents_sol == 0:
            return []
        agent_paths = [[] for _ in range(num_agents_sol)]
        for config in solution:
            for agent_id in range(num_agents_sol):
                y, x = config[agent_id]
                agent_paths[agent_id].append((x, y))
        return agent_paths
    
    paths1 = extract_paths(solution1, num_agents1) if has_solution1 else []
    paths2 = extract_paths(solution2, num_agents2) if has_solution2 else []
    paths3 = extract_paths(solution3, num_agents3) if has_solution3 else []
    
    # Initialize static plots (background)
    plot_solution_on_axis(ax1, solution1, title1, num_agents1)
    plot_solution_on_axis(ax2, solution2, title2, num_agents2)
    plot_solution_on_axis(ax3, solution3, title3, num_agents3)
    
    # Create agent position markers and path lines for animations
    agent_scatters1 = []
    path_lines1 = []
    agent_scatters2 = []
    path_lines2 = []
    agent_scatters3 = []
    path_lines3 = []
    
    # Initialize animation elements for solution 1
    if has_solution1 and paths1:
        for agent_id in range(num_agents1):
            scatter = ax1.scatter([], [], color=colors[agent_id % len(colors)],
                                 marker='o', s=250, edgecolors='black', linewidths=2,
                                 zorder=7)
            agent_scatters1.append(scatter)
            line, = ax1.plot([], [], color=colors[agent_id % len(colors)],
                            linewidth=2, alpha=0.5, zorder=6)
            path_lines1.append(line)
    
    # Initialize animation elements for solution 2
    if has_solution2 and paths2:
        for agent_id in range(num_agents2):
            scatter = ax2.scatter([], [], color=colors[agent_id % len(colors)],
                                 marker='o', s=250, edgecolors='black', linewidths=2,
                                 zorder=7)
            agent_scatters2.append(scatter)
            line, = ax2.plot([], [], color=colors[agent_id % len(colors)],
                            linewidth=2, alpha=0.5, zorder=6)
            path_lines2.append(line)
    
    # Initialize animation elements for solution 3
    if has_solution3 and paths3:
        for agent_id in range(num_agents3):
            scatter = ax3.scatter([], [], color=colors[agent_id % len(colors)],
                                 marker='o', s=250, edgecolors='black', linewidths=2,
                                 zorder=7)
            agent_scatters3.append(scatter)
            line, = ax3.plot([], [], color=colors[agent_id % len(colors)],
                            linewidth=2, alpha=0.5, zorder=6)
            path_lines3.append(line)
    
    # Time step text for each plot
    time_text1 = ax1.text(0.02, 0.02, '', transform=ax1.transAxes,
                         fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    time_text2 = ax2.text(0.02, 0.02, '', transform=ax2.transAxes,
                         fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    time_text3 = ax3.text(0.02, 0.02, '', transform=ax3.transAxes,
                         fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Function to update frame manually (for slider control only - no animation)
    def update_frame1(frame_val):
        frame = int(frame_val)
        if paths1:
            # Update frame directly
            time_text1.set_text(f'Time: {frame}/{len(solution1) - 1}')
            for agent_id in range(num_agents1):
                if frame < len(paths1[agent_id]):
                    x, y = paths1[agent_id][frame]
                    agent_scatters1[agent_id].set_offsets([[x, y]])
                    if frame > 0:
                        path_so_far = paths1[agent_id][:frame + 1]
                        xs, ys = zip(*path_so_far)
                        path_lines1[agent_id].set_data(xs, ys)
                    else:
                        path_lines1[agent_id].set_data([], [])
                else:
                    if paths1[agent_id]:
                        x, y = paths1[agent_id][-1]
                        agent_scatters1[agent_id].set_offsets([[x, y]])
            fig.canvas.draw_idle()
    
    def update_frame2(frame_val):
        frame = int(frame_val)
        if paths2:
            # Update frame directly
            time_text2.set_text(f'Time: {frame}/{len(solution2) - 1}')
            for agent_id in range(num_agents2):
                if frame < len(paths2[agent_id]):
                    x, y = paths2[agent_id][frame]
                    agent_scatters2[agent_id].set_offsets([[x, y]])
                    if frame > 0:
                        path_so_far = paths2[agent_id][:frame + 1]
                        xs, ys = zip(*path_so_far)
                        path_lines2[agent_id].set_data(xs, ys)
                    else:
                        path_lines2[agent_id].set_data([], [])
                else:
                    if paths2[agent_id]:
                        x, y = paths2[agent_id][-1]
                        agent_scatters2[agent_id].set_offsets([[x, y]])
            fig.canvas.draw_idle()
    
    def update_frame3(frame_val):
        frame = int(frame_val)
        if paths3:
            # Update frame directly
            time_text3.set_text(f'Time: {frame}/{len(solution3) - 1}')
            for agent_id in range(num_agents3):
                if frame < len(paths3[agent_id]):
                    x, y = paths3[agent_id][frame]
                    agent_scatters3[agent_id].set_offsets([[x, y]])
                    if frame > 0:
                        path_so_far = paths3[agent_id][:frame + 1]
                        xs, ys = zip(*path_so_far)
                        path_lines3[agent_id].set_data(xs, ys)
                    else:
                        path_lines3[agent_id].set_data([], [])
                else:
                    if paths3[agent_id]:
                        x, y = paths3[agent_id][-1]
                        agent_scatters3[agent_id].set_offsets([[x, y]])
            fig.canvas.draw_idle()
    
    # Add timeline sliders for each subplot (create before buttons and animations)
    max_frames1 = len(solution1) - 1 if has_solution1 and solution1 else 0
    max_frames2 = len(solution2) - 1 if has_solution2 and solution2 else 0
    max_frames3 = len(solution3) - 1 if has_solution3 and solution3 else 0
    
    # Create sliders below each subplot
    slider1 = Slider(plt.axes([0.125, 0.06, 0.22, 0.03]), 'Time', 
                     0, max(max_frames1, 1), valinit=0, valstep=1)
    slider1.on_changed(update_frame1)
    
    slider2 = Slider(plt.axes([0.45, 0.06, 0.22, 0.03]), 'Time',
                     0, max(max_frames2, 1), valinit=0, valstep=1)
    slider2.on_changed(update_frame2)
    
    slider3 = Slider(plt.axes([0.775, 0.06, 0.22, 0.03]), 'Time',
                     0, max(max_frames3, 1), valinit=0, valstep=1)
    slider3.on_changed(update_frame3)
    
    # Initialize frames to 0 when plot first loads
    if has_solution1 and paths1:
        update_frame1(0)
    
    if has_solution2 and paths2:
        update_frame2(0)
    
    if has_solution3 and paths3:
        update_frame3(0)
    
    # Add cost, SOC, and runtime information as text
    cost1 = len(solution1) - 1 if solution1 and len(solution1) > 0 else 0
    cost2 = len(solution2) - 1 if solution2 and len(solution2) > 0 else 0
    cost3 = len(solution3) - 1 if solution3 and len(solution3) > 0 else 0
    
    # Build info text for each plot
    info_text1 = f'Makespan: {cost1}'
    if soc1 is not None:
        info_text1 += f'\nSOC: {soc1}'
    if solution_time1 is not None:
        info_text1 += f'\nSolution found at: {solution_time1:.2f}ms'
    elif runtime1 is not None:
        info_text1 += f'\nTime: {runtime1:.2f}ms'
    
    info_text2 = f'Makespan: {cost2}'
    if soc2 is not None:
        info_text2 += f'\nSOC: {soc2}'
    if solution_time2 is not None:
        info_text2 += f'\nSolution found at: {solution_time2:.2f}ms'
    elif runtime2 is not None:
        info_text2 += f'\nTime: {runtime2:.2f}ms'
    
    info_text3 = f'Makespan: {cost3}'
    if soc3 is not None:
        info_text3 += f'\nSOC: {soc3}'
    if solution_time3 is not None:
        info_text3 += f'\nSolution found at: {solution_time3:.2f}ms'
    elif runtime3 is not None:
        info_text3 += f'\nTime: {runtime3:.2f}ms'
    
    if has_solution1:
        ax1.text(0.02, 0.98, info_text1, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    if has_solution2:
        ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    if has_solution3:
        ax3.text(0.02, 0.98, info_text3, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)  # Make room for sliders
    plt.show()
