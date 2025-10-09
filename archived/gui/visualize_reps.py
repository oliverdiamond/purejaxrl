#!/usr/bin/env python3
"""
Streamlit GUI for visualizing neural network activations in multi-task gridworld environments.

Usage:
    streamlit run visualize_reps.py -- <results_directory>

Example:
    streamlit run visualize_reps.py -- purejaxrl/results/dqn_multitask/TwoRoomsMultiTask5/MTQNet/25-01-29-10-30-45
"""

import pickle
import sys
import os
import io
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def load_data(results_dir: str) -> Tuple[Dict, Dict]:
    """Load activations and metadata from results directory."""
    try:
        # Look for activations.pkl in the results directory
        data_path = os.path.join(results_dir, "activations.pkl")
        
        if not os.path.exists(data_path):
            st.error(f"activations.pkl not found in {results_dir}")
            st.stop()
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['activations'], data['metadata']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


def create_observation_grid(grid_size: int, agent_loc: Tuple[int, int], 
                          goal_loc: Tuple[int, int], hallway_loc: Tuple[int, int]) -> np.ndarray:
    """Create a visual representation of the grid observation."""
    # Create base grid (white for open spaces)
    grid = np.ones((grid_size, grid_size, 3))  # RGB format
    
    # Add walls (black) - vertical wall at hallway column, except at hallway row
    for row in range(grid_size):
        if row != hallway_loc[0]:  # Not at hallway opening
            grid[row, hallway_loc[1]] = [0, 0, 0]  # Black
    
    # Add goal (green)
    grid[goal_loc[0], goal_loc[1]] = [0, 1, 0]  # Green
    
    # Add agent (blue)
    grid[agent_loc[0], agent_loc[1]] = [0, 0, 1]  # Blue
    
    return grid


def is_valid_location(row: int, col: int, grid_size: int, goal_loc: Tuple[int, int], 
                     hallway_loc: Tuple[int, int]) -> bool:
    """Check if a location is valid for the agent (not wall or goal)."""
    if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
        return False
    
    # Check if it's a wall
    is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
    
    # Check if it's the goal
    is_goal = (row == goal_loc[0] and col == goal_loc[1])
    
    return not (is_wall or is_goal)


def normalize_features(features: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """Normalize feature values to [0, 1] range using per-feature min/max values."""
    features = np.array(features)
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    
    # Handle cases where min == max for some features
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
    
    normalized = (features - min_vals) / range_vals
    # Set features with zero range to 0.5
    normalized = np.where(max_vals == min_vals, 0.5, normalized)
    
    return normalized


def create_feature_visualization(features: np.ndarray, title: str, 
                                feature_names: Optional[List[str]] = None,
                                raw_features: Optional[np.ndarray] = None,
                                min_vals: Optional[np.ndarray] = None,
                                max_vals: Optional[np.ndarray] = None) -> Figure:
    """Create a horizontal bar visualization of feature activations."""
    # Use raw features for display text, normalized features for colors
    display_features = raw_features if raw_features is not None else features
    # Create custom colormap: blue -> white -> red
    colors = ['blue', 'white', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('feature_map', colors, N=n_bins)
    
    # Fixed width of 10 inches, calculate square size to maintain aspect ratio
    fig_width = 10.0
    square_size = 2*(fig_width / len(features))  # Calculate square size based on number of features
    fig_height = square_size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    # Font size based on square size, but clamped for readability
    fontsize = max(8, min(16, square_size * 6))
    
    # Create rectangles for each feature
    for i, (norm_val, display_val) in enumerate(zip(features, display_features)):
        # Check if this feature has zero range (constant value)
        has_zero_range = False
        if min_vals is not None and max_vals is not None:
            has_zero_range = (min_vals[i] == max_vals[i])
        
        if has_zero_range:
            # Use yellow color for features with zero range
            color = 'yellow'
        else:
            # Use normalized value for color mapping
            color = cmap(norm_val)
        
        rect = patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.2)
        ax.add_patch(rect)
        
        # Add feature value text (display raw value)
        # Use black text for yellow background, otherwise use original logic
        if has_zero_range:
            text_color = 'black'
        else:
            text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
        ax.text(i + 0.5, 0.5, f'{display_val:.2f}', ha='center', va='center', 
                fontsize=fontsize, color=text_color, weight='bold')
    
    ax.set_xlim(0, len(features))
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xticks(np.arange(len(features)) + 0.5)
    
    if feature_names:
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    else:
        # Show fewer labels if many features
        if len(features) > 10:
            step = max(1, len(features) // 8)
            tick_indices = range(0, len(features), step)
            ax.set_xticks([i + 0.5 for i in tick_indices])
            ax.set_xticklabels([f'{i}' for i in tick_indices], fontsize=6)
        else:
            ax.set_xticklabels([f'{i}' for i in range(len(features))], fontsize=7)
    
    ax.set_yticks([])
    #ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def create_combined_visualization(activation_data: Dict[str, List[np.ndarray]], 
                                activation_titles: Dict[str, List[str]],
                                grid_thumbnails: Optional[Dict[str, List[np.ndarray]]] = None,
                                raw_activation_data: Optional[Dict[str, List[np.ndarray]]] = None,
                                normalization_ranges: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> Dict[str, Figure]:
    """Create separate plots for each activation type with shape (n_observations, n_features)."""
    if not activation_data:
        return {}
    
    # Create custom colormap
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('feature_map', colors, N=256)
    
    figures = {}
    
    for activation_type, features_list in activation_data.items():
        if not features_list:
            continue
            
        # Stack all features of this type
        stacked_features = np.array(features_list)  # Shape: (n_observations, n_features)
        n_obs, n_features = stacked_features.shape
        
        # Get raw features for display if available
        if raw_activation_data and activation_type in raw_activation_data:
            stacked_raw_features = np.array(raw_activation_data[activation_type])
        else:
            stacked_raw_features = stacked_features
        
        # Get normalization ranges for zero-range detection
        min_vals = None
        max_vals = None
        if normalization_ranges and activation_type in normalization_ranges:
            min_vals = normalization_ranges[activation_type]['min']
            max_vals = normalization_ranges[activation_type]['max']
        
        # Create figure with space for thumbnails on the left - made much larger
        thumbnail_width = 1.2 if grid_thumbnails else 0  # Increased thumbnail space
        fig_width = max(16, n_features * 0.8) + thumbnail_width  # Much wider
        fig_height = max(8, n_obs * 1.2)  # Much taller
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)  # Higher DPI
        
        # Adjust subplot to make room for thumbnails
        if grid_thumbnails:
            plt.subplots_adjust(left=0.08)  # Adjusted for larger thumbnails
        
        # Create heatmap (offset by thumbnail width if needed)
        x_offset = thumbnail_width * 1.5 if grid_thumbnails else 0  # Increased offset
        
        for obs_idx in range(n_obs):
            for feat_idx in range(n_features):
                norm_val = stacked_features[obs_idx, feat_idx]  # Normalized value for color
                raw_val = stacked_raw_features[obs_idx, feat_idx]  # Raw value for display
                
                # Check if this feature has zero range (constant value)
                has_zero_range = False
                if min_vals is not None and max_vals is not None:
                    has_zero_range = (min_vals[feat_idx] == max_vals[feat_idx])
                
                if has_zero_range:
                    # Use yellow color for features with zero range
                    color = 'yellow'
                else:
                    # Use normalized value for color mapping
                    color = cmap(norm_val)
                
                rect = patches.Rectangle((feat_idx + x_offset, n_obs - obs_idx - 1), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=0.3)
                ax.add_patch(rect)
                
                # Add feature value text (display raw value)
                # Use black text for yellow background, otherwise use original logic
                if has_zero_range:
                    text_color = 'black'
                else:
                    text_color = 'white' if norm_val < 0.3 or norm_val > 0.7 else 'black'
                fontsize = min(12, max(8, 120 / max(n_obs, n_features)))  # Larger font for bigger plots
                ax.text(feat_idx + x_offset + 0.5, n_obs - obs_idx - 0.5, f'{raw_val:.2f}', 
                       ha='center', va='center', fontsize=fontsize, 
                       color=text_color, weight='bold')
        
        # Add grid thumbnails if provided
        if grid_thumbnails and activation_type in grid_thumbnails:
            thumbnails = grid_thumbnails[activation_type]
            for obs_idx, thumbnail in enumerate(thumbnails):
                if thumbnail is not None:
                    # Create mini thumbnail
                    y_pos = n_obs - obs_idx - 1
                    
                    # Add thumbnail as a small image
                    thumbnail_size = 0.8
                    extent = (-0.1 - thumbnail_size, -0.1, y_pos + 0.1, y_pos + 0.9)
                    ax.imshow(thumbnail, extent=extent, aspect='equal')
                    
                    # Add border around thumbnail
                    rect = patches.Rectangle((-0.1 - thumbnail_size, y_pos + 0.1), 
                                           thumbnail_size, thumbnail_size, 
                                           linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
        
        ax.set_xlim(-0.2 - thumbnail_width if grid_thumbnails else 0, n_features + x_offset)
        ax.set_ylim(0, n_obs)
        ax.set_title(f'{activation_type} - ({n_obs} observations)', fontsize=12, pad=10)
        
        # Set y-axis labels (observation names)
        ax.set_yticks(np.arange(n_obs) + 0.5)
        if activation_type in activation_titles:
            ax.set_yticklabels(activation_titles[activation_type][::-1])  # Reverse for proper display
        else:
            ax.set_yticklabels([f'Obs {i+1}' for i in range(n_obs)][::-1])
        
        # Set x-axis labels (feature indices)
        ax.set_xticks(np.arange(n_features) + 0.5 + x_offset)
        ax.set_xticklabels([f'F{i}' for i in range(n_features)], rotation=45, ha='right')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        
        figures[activation_type] = fig
    
    return figures


def main():
    st.set_page_config(page_title="Neural Activation Visualizer", layout="wide")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        st.error("Please provide a results directory as argument: streamlit run script.py -- results_directory")
        st.stop()
    
    st.title("Activation Heat Maps For Multi-Task DQN Architecture in Gridworld Maze")
    st.markdown("---")
    
    # Load data
    activations, metadata = load_data(results_dir)
    
    # Display metadata
    with st.expander("Experiment Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Metadata")
            st.write(f"**Grid Size:** {metadata['grid_size']}×{metadata['grid_size']}")
            
            # Determine number of tasks based on environment type
            config = metadata['config']
            hallway_is_task = "HallwayAsTask" in config.get("ENV_NAME", "")
            num_tasks = metadata['num_hallways'] if hallway_is_task else metadata['num_goals']
            st.write(f"**Number of Tasks:** {num_tasks}")
            
            st.write(f"**Goal Locations:** {metadata['goal_locs']}")
            st.write(f"**Start Location:** {metadata['start_loc']}")
            st.write(f"**Hallway Locations:** {metadata['hallway_locs']}")
        
        with col2:
            st.subheader("Experiment Configuration")
            config = metadata['config']
            
            # Display full config in a structured way
            for key, value in config.items():
                # Format the key to be more readable
                formatted_key = key.replace('_', ' ').title()
                st.write(f"**{formatted_key}:** {value}")
    
    # Initialize session state
    if 'grid_configs' not in st.session_state:
        # Default: one grid for each (goal, hallway) combination
        st.session_state.grid_configs = []
        if metadata['num_goals'] > 0 and metadata['num_hallways'] > 0:
            start_loc = tuple(metadata['start_loc'])
            # Create a grid for each goal-hallway combination
            for goal_idx in range(metadata['num_goals']):
                for hallway_idx in range(metadata['num_hallways']):
                    st.session_state.grid_configs.append({
                        'goal_idx': goal_idx,
                        'hallway_idx': hallway_idx,
                        'agent_loc': start_loc
                    })
    
    # Determine if network has task-specific representations
    config = metadata['config']
    network_name = config.get('NETWORK_NAME', '')
    has_task_reps = network_name == 'MTQNetWithTaskReps'
    
    # Control panel
    st.subheader("Controls")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Visualization options
        st.write("**Activation Displays:**")
        show_shared_rep = st.checkbox("Show Shared Representations", value=True)
        
        if has_task_reps:
            show_task_rep = st.checkbox("Show Task Representations", value=True)
            if show_task_rep:
                # Determine number of tasks based on environment type
                config = metadata['config']
                hallway_is_task = "HallwayAsTask" in config.get("ENV_NAME", "")
                num_tasks = metadata['num_hallways'] if hallway_is_task else metadata['num_goals']

                selected_tasks = st.multiselect(
                    "Select Task Representations",
                    options=list(range(num_tasks)),
                    default=[0] if num_tasks > 0 else []
                )
            else:
                selected_tasks = []
        else:
            show_task_rep = False
            selected_tasks = []
    
    with col2:
        # Display mode
        display_mode = st.radio(
            "**Display Mode:**",
            ["Separate (next to each grid)", "Combined (grouped by type)"],
            index=0
        )
        
        # Grid management
        st.write("**Grid Management:**")
        if st.button("➕ Add Grid"):
            st.session_state.grid_configs.append({
                'goal_idx': 0,
                'hallway_idx': 0,
                'agent_loc': tuple(metadata['start_loc'])
            })
            st.rerun()
    
    with col3:
        if len(st.session_state.grid_configs) > 1 and st.button("➖ Remove Last Grid"):
            st.session_state.grid_configs.pop()
            st.rerun()
    
    st.markdown("---")
    
    # Compute normalization ranges - per feature normalization
    all_shared_reps = []
    all_task_reps = {}
    
    # Determine number of tasks based on environment type
    config = metadata['config']
    hallway_is_task = "HallwayAsTask" in config.get("ENV_NAME", "")
    num_tasks = metadata['num_hallways'] if hallway_is_task else metadata['num_goals']
    
    if has_task_reps:
        all_task_reps = {task_idx: [] for task_idx in range(num_tasks)}
    
    # Iterate through all goal-hallway combinations
    for goal_idx in range(metadata['num_goals']):
        for hallway_idx in range(metadata['num_hallways']):
            combo_key = (goal_idx, hallway_idx)
            if combo_key in activations:
                combo_data = activations[combo_key]
                
                # Collect shared representations
                for (row, col), shared_rep in combo_data['shared_rep'].items():
                    all_shared_reps.append(shared_rep)
                
                # Collect task representations if available
                if has_task_reps and 'task_rep' in combo_data:
                    for (row, col), task_rep_array in combo_data['task_rep'].items():
                        # Determine which task to use based on environment type
                        if hallway_is_task:
                            task_idx = hallway_idx  # Task defined by hallway
                        else:
                            task_idx = goal_idx     # Task defined by goal
                        
                        if task_idx < len(task_rep_array):
                            all_task_reps[task_idx].append(task_rep_array[task_idx])
    
    # Compute per-feature normalization ranges for shared representations
    if all_shared_reps:
        shared_reps_array = np.array(all_shared_reps)  # Shape: (n_observations, n_features)
        shared_rep_min = np.min(shared_reps_array, axis=0)  # Per-feature min
        shared_rep_max = np.max(shared_reps_array, axis=0)  # Per-feature max
    else:
        shared_rep_min = np.array([0])
        shared_rep_max = np.array([1])
    
    # Compute per-feature normalization ranges for task representations
    task_rep_ranges = {}
    if has_task_reps:
        for task_idx in range(num_tasks):
            if task_idx in all_task_reps and all_task_reps[task_idx]:
                task_reps_array = np.array(all_task_reps[task_idx])  # Shape: (n_observations, n_features)
                task_rep_ranges[task_idx] = {
                    'min': np.min(task_reps_array, axis=0),  # Per-feature min
                    'max': np.max(task_reps_array, axis=0)   # Per-feature max
                }
            else:
                task_rep_ranges[task_idx] = {'min': np.array([0]), 'max': np.array([1])}
    
    # Main visualization
    st.subheader("Observations and Activations")
    
    # Collect data for combined view
    combined_activation_data = {
        'Shared Representations': []
    }
    combined_activation_titles = {
        'Shared Representations': []
    }
    combined_grid_thumbnails = {
        'Shared Representations': []
    }
    
    # Collect raw data for display
    raw_activation_data = {
        'Shared Representations': []
    }
    
    # Add task-specific data structures if available
    if has_task_reps:
        for task_idx in range(num_tasks):
            combined_activation_data[f'Task {task_idx} Representations'] = []
            combined_activation_titles[f'Task {task_idx} Representations'] = []
            combined_grid_thumbnails[f'Task {task_idx} Representations'] = []
            raw_activation_data[f'Task {task_idx} Representations'] = []
    
    if display_mode == "Separate (next to each grid)":
        # Show each grid with its activations side-by-side
        for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
            st.write(f"**Grid {grid_idx + 1}**")
            
            # Create side-by-side layout for this grid
            grid_col, activation_col = st.columns([0.5, 2])
            
            with grid_col:
                # Goal selection
                new_goal_idx = st.selectbox(
                    f"Goal",
                    options=list(range(metadata['num_goals'])),
                    index=grid_config['goal_idx'],
                    key=f"goal_{grid_idx}"
                )
                
                # Hallway selection
                new_hallway_idx = st.selectbox(
                    f"Hallway",
                    options=list(range(metadata['num_hallways'])),
                    index=grid_config['hallway_idx'],
                    key=f"hallway_{grid_idx}"
                )
                
                if new_goal_idx != grid_config['goal_idx'] or new_hallway_idx != grid_config['hallway_idx']:
                    st.session_state.grid_configs[grid_idx]['goal_idx'] = new_goal_idx
                    st.session_state.grid_configs[grid_idx]['hallway_idx'] = new_hallway_idx
                    st.rerun()
                
                # Create and display observation grid (smaller size)
                goal_idx = grid_config['goal_idx']
                hallway_idx = grid_config['hallway_idx']
                goal_loc = tuple(metadata['goal_locs'][goal_idx])
                hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
                agent_loc = grid_config['agent_loc']
                
                # Create observation visualization
                grid_vis = create_observation_grid(
                    metadata['grid_size'], agent_loc, goal_loc, hallway_loc
                )
                
                fig, ax = plt.subplots(figsize=(1.8, 1.8))  # Smaller grid size
                ax.imshow(grid_vis)
                ax.set_xticks(range(metadata['grid_size']))
                ax.set_yticks(range(metadata['grid_size']))
                ax.tick_params(labelsize=6)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Agent location controls (more compact)
                col1, col2 = st.columns(2)
                with col1:
                    new_row = st.number_input(f"Row", 
                                            min_value=0, max_value=metadata['grid_size']-1, 
                                            value=agent_loc[0], key=f"row_{grid_idx}")
                with col2:
                    new_col = st.number_input(f"Col", 
                                            min_value=0, max_value=metadata['grid_size']-1, 
                                            value=agent_loc[1], key=f"col_{grid_idx}")
                
                if (new_row, new_col) != agent_loc:
                    if is_valid_location(new_row, new_col, metadata['grid_size'], goal_loc, hallway_loc):
                        st.session_state.grid_configs[grid_idx]['agent_loc'] = (new_row, new_col)
                        st.rerun()
                    else:
                        st.warning("Invalid location!")
            
            with activation_col:
                # Show activations for this specific grid
                combo_key = (goal_idx, hallway_idx)
                if combo_key in activations and agent_loc in activations[combo_key]['shared_rep']:
                    combo_data = activations[combo_key]
                    shared_rep = combo_data['shared_rep'][agent_loc]
                    
                    # Normalize features
                    normalized_shared = normalize_features(shared_rep, shared_rep_min, shared_rep_max)
                    
                    if show_shared_rep:
                        fig = create_feature_visualization(
                            normalized_shared, 
                            f"Shared Rep",
                            raw_features=shared_rep,
                            min_vals=shared_rep_min,
                            max_vals=shared_rep_max
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    if show_task_rep and has_task_reps and 'task_rep' in combo_data:
                        task_rep_array = combo_data['task_rep'][agent_loc]
                        for task_idx in selected_tasks:
                            if task_idx < len(task_rep_array):
                                task_rep = task_rep_array[task_idx]
                                task_range = task_rep_ranges[task_idx]
                                normalized_task = normalize_features(task_rep, task_range['min'], task_range['max'])
                                
                                fig = create_feature_visualization(
                                    normalized_task,
                                    f"Task {task_idx} Rep",
                                    raw_features=task_rep,
                                    min_vals=task_range['min'],
                                    max_vals=task_range['max']
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                else:
                    st.warning(f"No activation data available for goal {goal_idx}, hallway {hallway_idx}, agent at {agent_loc}")
            
            st.markdown("---")
    
    else:  # Combined view - collect all data first, then show combined plots
        # First, collect all the activation data
        for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
            goal_idx = grid_config['goal_idx']
            hallway_idx = grid_config['hallway_idx']
            goal_loc = tuple(metadata['goal_locs'][goal_idx])
            hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
            agent_loc = grid_config['agent_loc']
            
            # Create grid thumbnail for this configuration
            grid_thumbnail = create_observation_grid(
                metadata['grid_size'], agent_loc, goal_loc, hallway_loc
            )
            
            # Collect activation data for this grid
            combo_key = (goal_idx, hallway_idx)
            if combo_key in activations and agent_loc in activations[combo_key]['shared_rep']:
                combo_data = activations[combo_key]
                shared_rep = combo_data['shared_rep'][agent_loc]
                
                # Normalize features
                normalized_shared = normalize_features(shared_rep, shared_rep_min, shared_rep_max)
                
                # Store for combined view
                if show_shared_rep:
                    combined_activation_data['Shared Representations'].append(normalized_shared)
                    combined_activation_titles['Shared Representations'].append(f"Grid {grid_idx + 1}")
                    combined_grid_thumbnails['Shared Representations'].append(grid_thumbnail)
                    raw_activation_data['Shared Representations'].append(shared_rep)
                
                if show_task_rep and has_task_reps and 'task_rep' in combo_data:
                    task_rep_array = combo_data['task_rep'][agent_loc]
                    for task_idx in selected_tasks:
                        if task_idx < len(task_rep_array):
                            task_rep = task_rep_array[task_idx]
                            task_range = task_rep_ranges[task_idx]
                            normalized_task = normalize_features(task_rep, task_range['min'], task_range['max'])
                            combined_activation_data[f'Task {task_idx} Representations'].append(normalized_task)
                            combined_activation_titles[f'Task {task_idx} Representations'].append(f"Grid {grid_idx + 1}")
                            combined_grid_thumbnails[f'Task {task_idx} Representations'].append(grid_thumbnail)
                            raw_activation_data[f'Task {task_idx} Representations'].append(task_rep)
        
        # Show all grids in a compact view first
        st.write("**All Observations**")
        
        # Create columns for grids (show multiple per row)
        grids_per_row = min(9, len(st.session_state.grid_configs))  # Maximum 6 columns
        if grids_per_row > 0:
            grid_cols = st.columns(grids_per_row)
            
            for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
                col_idx = grid_idx % grids_per_row
                with grid_cols[col_idx]:
                    # Goal selection (compact)
                    new_goal_idx = st.selectbox(
                        f"G{grid_idx + 1}",
                        options=list(range(metadata['num_goals'])),
                        index=grid_config['goal_idx'],
                        key=f"comb_goal_{grid_idx}"
                    )
                    
                    # Hallway selection (compact)
                    new_hallway_idx = st.selectbox(
                        f"H{grid_idx + 1}",
                        options=list(range(metadata['num_hallways'])),
                        index=grid_config['hallway_idx'],
                        key=f"comb_hallway_{grid_idx}"
                    )
                    
                    if new_goal_idx != grid_config['goal_idx'] or new_hallway_idx != grid_config['hallway_idx']:
                        st.session_state.grid_configs[grid_idx]['goal_idx'] = new_goal_idx
                        st.session_state.grid_configs[grid_idx]['hallway_idx'] = new_hallway_idx
                        st.rerun()
                    
                    # Create and display observation grid (very small)
                    goal_idx = grid_config['goal_idx']
                    hallway_idx = grid_config['hallway_idx']
                    goal_loc = tuple(metadata['goal_locs'][goal_idx])
                    hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
                    agent_loc = grid_config['agent_loc']
                    
                    # Create observation visualization
                    grid_vis = create_observation_grid(
                        metadata['grid_size'], agent_loc, goal_loc, hallway_loc
                    )
                    
                    fig, ax = plt.subplots(figsize=(1.5, 1.5))  # Very small for combined view
                    ax.imshow(grid_vis)
                    ax.set_xticks(range(metadata['grid_size']))
                    ax.set_yticks(range(metadata['grid_size']))
                    ax.tick_params(labelsize=5)
                    
                    st.pyplot(fig)
                    
                    plt.close(fig)
                    
                    # Agent location controls (very compact)
                    r_col, c_col = st.columns(2)
                    with r_col:
                        new_row = st.number_input(f"R", 
                                                min_value=0, max_value=metadata['grid_size']-1, 
                                                value=agent_loc[0], key=f"comb_row_{grid_idx}")
                    with c_col:
                        new_col = st.number_input(f"C", 
                                                min_value=0, max_value=metadata['grid_size']-1, 
                                                value=agent_loc[1], key=f"comb_col_{grid_idx}")
                    
                    if (new_row, new_col) != agent_loc:
                        if is_valid_location(new_row, new_col, metadata['grid_size'], goal_loc, hallway_loc):
                            st.session_state.grid_configs[grid_idx]['agent_loc'] = (new_row, new_col)
                            st.rerun()
                        else:
                            st.warning("Invalid!")
        
        st.markdown("---")
        st.write("**Combined Activation View**")
        
        # Prepare normalization ranges for combined visualization
        normalization_ranges = {
            'Shared Representations': {'min': shared_rep_min, 'max': shared_rep_max}
        }
        
        # Add task-specific normalization ranges if available
        if has_task_reps:
            for task_idx in range(num_tasks):
                if task_idx in task_rep_ranges:
                    normalization_ranges[f'Task {task_idx} Representations'] = {
                        'min': task_rep_ranges[task_idx]['min'],
                        'max': task_rep_ranges[task_idx]['max']
                    }
        
        # Create combined visualizations
        combined_figures = create_combined_visualization(
            combined_activation_data, 
            combined_activation_titles, 
            combined_grid_thumbnails,
            raw_activation_data,
            normalization_ranges
        )
        
        for activation_type, fig in combined_figures.items():
            if activation_type == 'Shared Representations' and show_shared_rep:
                plt.savefig(f"{results_dir}/shared_rep_combined.png")
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            elif activation_type.startswith('Task') and show_task_rep:
                task_idx = int(activation_type.split()[1])
                if task_idx in selected_tasks:
                    plt.savefig(f"{results_dir}/task_{task_idx}_rep_combined.png")
                    st.pyplot(fig)
                    plt.close(fig)


if __name__ == "__main__":
    main()
