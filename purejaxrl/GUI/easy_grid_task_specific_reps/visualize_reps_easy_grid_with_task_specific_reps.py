#!/usr/bin/env python3
"""
Streamlit GUI for visualizing neural network activations in multi-task gridworld environments.

Usage:
    streamlit run visualize_reps_easy_grid_with_task_specific_reps.py -- <filename>

Example:
    streamlit run visualize_reps_easy_grid_with_task_specific_reps.py -- activations_N=5_n_tasks=3_shared_bottleneck=8_task_features=32_conv1=32_conv2=16_seed=0.pkl
"""

import pickle
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def load_data(filename: str) -> Tuple[Dict, Dict]:
    """Load activations and metadata from pickle file."""
    try:
        # Navigate up to the project root to find the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        data_path = os.path.join(project_root, "purejaxrl", "data", filename)
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data['activations'], data['metadata']
    except FileNotFoundError:
        st.error(f"File {filename} not found in data directory!")
        st.error(f"Looking for: {data_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
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


def normalize_features(features: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalize feature values to [0, 1] range."""
    if max_val == min_val:
        return np.full_like(features, 0.5)
    return (features - min_val) / (max_val - min_val)


def create_feature_visualization(features: np.ndarray, title: str, 
                                feature_names: Optional[List[str]] = None, compact: bool = False) -> Figure:
    """Create a horizontal bar visualization of feature activations."""
    # Create custom colormap: blue -> white -> red
    colors = ['blue', 'white', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('feature_map', colors, N=n_bins)
    
    # Adjust size based on compact mode and feature type
    if compact:
        # Make Q-values (4 features) and shared rep much smaller in separate mode
        if len(features) <= 8:  # Q-values or small shared rep
            fig, ax = plt.subplots(figsize=(max(2, len(features) * 0.25), 0.6))
            fontsize = max(5, min(8, 25 / len(features)))
        else:  # Larger feature arrays
            fig, ax = plt.subplots(figsize=(max(3, len(features) * 0.15), 0.8))
            fontsize = max(4, min(6, 30 / len(features)))
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.5), 2))
        fontsize = 8
    
    # Create rectangles for each feature
    for i, val in enumerate(features):
        color = cmap(val)
        rect = patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.2)
        ax.add_patch(rect)
        
        # Add feature value text
        text_color = 'white' if val < 0.3 or val > 0.7 else 'black'
        ax.text(i + 0.5, 0.5, f'{val:.2f}', ha='center', va='center', 
                fontsize=fontsize, color=text_color, weight='bold')
    
    ax.set_xlim(0, len(features))
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9 if compact else 12, pad=6)
    ax.set_xticks(np.arange(len(features)) + 0.5)
    
    if feature_names:
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    else:
        # Show fewer labels if compact and many features
        if compact and len(features) > 10:
            step = max(1, len(features) // 8)
            tick_indices = range(0, len(features), step)
            ax.set_xticks([i + 0.5 for i in tick_indices])
            ax.set_xticklabels([f'F{i}' for i in tick_indices], fontsize=6)
        else:
            ax.set_xticklabels([f'F{i}' for i in range(len(features))], fontsize=7)
    
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def create_combined_visualization(activation_data: Dict[str, List[np.ndarray]], 
                                activation_titles: Dict[str, List[str]],
                                grid_thumbnails: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, Figure]:
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
        
        # Create figure with space for thumbnails on the left
        thumbnail_width = 0.8 if grid_thumbnails else 0
        fig_width = max(8, n_features * 0.3) + thumbnail_width
        fig, ax = plt.subplots(figsize=(fig_width, max(4, n_obs * 0.5)))
        
        # Adjust subplot to make room for thumbnails
        if grid_thumbnails:
            plt.subplots_adjust(left=0.12)
        
        # Create heatmap (offset by thumbnail width if needed)
        x_offset = thumbnail_width * 1.2 if grid_thumbnails else 0
        
        for obs_idx in range(n_obs):
            for feat_idx in range(n_features):
                val = stacked_features[obs_idx, feat_idx]
                color = cmap(val)
                rect = patches.Rectangle((feat_idx + x_offset, n_obs - obs_idx - 1), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=0.3)
                ax.add_patch(rect)
                
                # Add feature value text (smaller font for dense plots)
                text_color = 'white' if val < 0.3 or val > 0.7 else 'black'
                fontsize = min(8, max(4, 60 / max(n_obs, n_features)))
                ax.text(feat_idx + x_offset + 0.5, n_obs - obs_idx - 0.5, f'{val:.2f}', 
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
        ax.set_title(f'{activation_type} - Combined View ({n_obs} observations)', fontsize=12, pad=10)
        
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
        filename = sys.argv[1]
    else:
        st.error("Please provide a filename as argument: streamlit run script.py -- filename.pkl")
        st.stop()
    
    st.title("🧠 Multi-Task Gridworld Neural Activation Visualizer")
    st.markdown("---")
    
    # Load data
    activations, metadata = load_data(filename)
    
    # Display metadata
    with st.expander("📊 Dataset Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Metadata")
            st.write(f"**Grid Size:** {metadata['grid_size']}×{metadata['grid_size']}")
            st.write(f"**Number of Tasks:** {metadata['num_hallways']}")
            st.write(f"**Goal Location:** {metadata['goal_loc']}")
            st.write(f"**Start Location:** {metadata['start_loc']}")
            st.write(f"**Hallway Locations:** {metadata['hallway_locs']}")
        
        with col2:
            st.subheader("Model Configuration")
            config = metadata['config']
            st.write(f"**Shared Bottleneck:** {config.get('N_SHARED_BOTTLENECK', 'N/A')}")
            st.write(f"**Task Features:** {config.get('N_FEATURES_PER_TASK', 'N/A')}")
            st.write(f"**Conv1 Features:** {config.get('N_FEATURES_CONV1', 'N/A')}")
            st.write(f"**Conv2 Features:** {config.get('N_FEATURES_CONV2', 'N/A')}")
            st.write(f"**Learning Rate:** {config.get('LR', 'N/A')}")
            st.write(f"**Seed:** {config.get('SEED', 'N/A')}")
    
    # Initialize session state
    if 'grid_configs' not in st.session_state:
        # Default: one grid per hallway
        st.session_state.grid_configs = []
        for i in range(metadata['num_hallways']):
            hallway_loc = tuple(metadata['hallway_locs'][i])
            start_loc = tuple(metadata['start_loc'])
            st.session_state.grid_configs.append({
                'hallway_idx': i,
                'agent_loc': start_loc
            })
    
    # Control panel
    st.subheader("🎛️ Control Panel")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Visualization options
        st.write("**Activation Displays:**")
        show_shared_rep = st.checkbox("Show Shared Representations", value=True)
        show_task_rep = st.checkbox("Show Task Representations", value=True)
        show_q_values = st.checkbox("Show Q-Values", value=False)
        
        if show_task_rep:
            selected_tasks = st.multiselect(
                "Select Task Representations",
                options=list(range(metadata['num_hallways'])),
                default=[0] if metadata['num_hallways'] > 0 else []
            )
        else:
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
                'hallway_idx': 0,
                'agent_loc': tuple(metadata['start_loc'])
            })
            st.rerun()
    
    with col3:
        if len(st.session_state.grid_configs) > 1 and st.button("➖ Remove Last Grid"):
            st.session_state.grid_configs.pop()
            st.rerun()
    
    st.markdown("---")
    
    # Compute normalization ranges
    all_shared_reps = []
    all_task_reps = {task_idx: [] for task_idx in range(metadata['num_hallways'])}
    
    for hallway_idx in range(metadata['num_hallways']):
        for (row, col), shared_rep in activations[hallway_idx]['shared_rep'].items():
            all_shared_reps.append(shared_rep)
        
        for (row, col), task_rep_array in activations[hallway_idx]['task_rep'].items():
            for task_idx in range(metadata['num_hallways']):
                all_task_reps[task_idx].append(task_rep_array[task_idx])
    
    # Compute normalization ranges
    shared_rep_min = np.min([np.min(rep) for rep in all_shared_reps]) if all_shared_reps else 0
    shared_rep_max = np.max([np.max(rep) for rep in all_shared_reps]) if all_shared_reps else 1
    
    task_rep_ranges = {}
    for task_idx in range(metadata['num_hallways']):
        if all_task_reps[task_idx]:
            task_rep_ranges[task_idx] = {
                'min': np.min([np.min(rep) for rep in all_task_reps[task_idx]]),
                'max': np.max([np.max(rep) for rep in all_task_reps[task_idx]])
            }
        else:
            task_rep_ranges[task_idx] = {'min': 0, 'max': 1}
    
    # Main visualization
    st.subheader("🔍 Grid Observations and Neural Activations")
    
    # Collect data for combined view
    combined_activation_data = {
        'Shared Representations': [],
        'Q-Values': []
    }
    combined_activation_titles = {
        'Shared Representations': [],
        'Q-Values': []
    }
    combined_grid_thumbnails = {
        'Shared Representations': [],
        'Q-Values': []
    }
    
    # Add task-specific data structures
    for task_idx in range(metadata['num_hallways']):
        combined_activation_data[f'Task {task_idx} Representations'] = []
        combined_activation_titles[f'Task {task_idx} Representations'] = []
        combined_grid_thumbnails[f'Task {task_idx} Representations'] = []
    
    if display_mode == "Separate (next to each grid)":
        # Show each grid with its activations side-by-side
        for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
            st.write(f"**Grid {grid_idx + 1}**")
            
            # Create side-by-side layout for this grid
            grid_col, activation_col = st.columns([1, 2])
            
            with grid_col:
                # Hallway selection (compact)
                new_hallway_idx = st.selectbox(
                    f"Hallway",
                    options=list(range(metadata['num_hallways'])),
                    index=grid_config['hallway_idx'],
                    key=f"hallway_{grid_idx}"
                )
                
                if new_hallway_idx != grid_config['hallway_idx']:
                    st.session_state.grid_configs[grid_idx]['hallway_idx'] = new_hallway_idx
                    st.rerun()
                
                # Create and display observation grid (smaller size)
                hallway_idx = grid_config['hallway_idx']
                hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
                agent_loc = grid_config['agent_loc']
                goal_loc = tuple(metadata['goal_loc'])
                
                # Create observation visualization
                grid_vis = create_observation_grid(
                    metadata['grid_size'], agent_loc, goal_loc, hallway_loc
                )
                
                fig, ax = plt.subplots(figsize=(2.5, 2.5))  # Even smaller grid size
                ax.imshow(grid_vis)
                ax.set_title(f'H{hallway_idx} at {hallway_loc}', fontsize=9)
                ax.set_xticks(range(metadata['grid_size']))
                ax.set_yticks(range(metadata['grid_size']))
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
                
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
                if agent_loc in activations[hallway_idx]['shared_rep']:
                    shared_rep = activations[hallway_idx]['shared_rep'][agent_loc]
                    task_rep_array = activations[hallway_idx]['task_rep'][agent_loc]
                    q_values_array = activations[hallway_idx]['q_values'][agent_loc]
                    
                    # Normalize features
                    normalized_shared = normalize_features(shared_rep, shared_rep_min, shared_rep_max)
                    
                    if show_shared_rep:
                        fig = create_feature_visualization(
                            normalized_shared, 
                            f"Shared Rep",
                            compact=True
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    if show_task_rep:
                        for task_idx in selected_tasks:
                            task_rep = task_rep_array[task_idx]
                            task_range = task_rep_ranges[task_idx]
                            normalized_task = normalize_features(task_rep, task_range['min'], task_range['max'])
                            
                            fig = create_feature_visualization(
                                normalized_task,
                                f"Task {task_idx} Rep",
                                compact=True
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    if show_q_values:
                        q_values = q_values_array[hallway_idx]
                        normalized_q = normalize_features(q_values, np.min(q_values), np.max(q_values))
                        
                        fig = create_feature_visualization(
                            normalized_q,
                            f"Q-Values",
                            ['Up', 'Right', 'Down', 'Left'],
                            compact=True
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.warning(f"No activation data available")
            
            st.markdown("---")
    
    else:  # Combined view - collect all data first, then show combined plots
        # First, collect all the activation data
        for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
            hallway_idx = grid_config['hallway_idx']
            hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
            agent_loc = grid_config['agent_loc']
            goal_loc = tuple(metadata['goal_loc'])
            
            # Create grid thumbnail for this configuration
            grid_thumbnail = create_observation_grid(
                metadata['grid_size'], agent_loc, goal_loc, hallway_loc
            )
            
            # Collect activation data for this grid
            if agent_loc in activations[hallway_idx]['shared_rep']:
                shared_rep = activations[hallway_idx]['shared_rep'][agent_loc]
                task_rep_array = activations[hallway_idx]['task_rep'][agent_loc]
                q_values_array = activations[hallway_idx]['q_values'][agent_loc]
                
                # Normalize features
                normalized_shared = normalize_features(shared_rep, shared_rep_min, shared_rep_max)
                
                # Store for combined view
                if show_shared_rep:
                    combined_activation_data['Shared Representations'].append(normalized_shared)
                    combined_activation_titles['Shared Representations'].append(f"Grid {grid_idx + 1}")
                    combined_grid_thumbnails['Shared Representations'].append(grid_thumbnail)
                
                if show_task_rep:
                    for task_idx in selected_tasks:
                        task_rep = task_rep_array[task_idx]
                        task_range = task_rep_ranges[task_idx]
                        normalized_task = normalize_features(task_rep, task_range['min'], task_range['max'])
                        combined_activation_data[f'Task {task_idx} Representations'].append(normalized_task)
                        combined_activation_titles[f'Task {task_idx} Representations'].append(f"Grid {grid_idx + 1}")
                        combined_grid_thumbnails[f'Task {task_idx} Representations'].append(grid_thumbnail)
                
                if show_q_values:
                    q_values = q_values_array[hallway_idx]
                    normalized_q = normalize_features(q_values, np.min(q_values), np.max(q_values))
                    combined_activation_data['Q-Values'].append(normalized_q)
                    combined_activation_titles['Q-Values'].append(f"Grid {grid_idx + 1}")
                    combined_grid_thumbnails['Q-Values'].append(grid_thumbnail)
        
        # Show all grids in a compact view first
        st.write("**All Observation Grids**")
        
        # Create columns for grids (show multiple per row)
        grids_per_row = min(4, len(st.session_state.grid_configs))
        if grids_per_row > 0:
            grid_cols = st.columns(grids_per_row)
            
            for grid_idx, grid_config in enumerate(st.session_state.grid_configs):
                col_idx = grid_idx % grids_per_row
                with grid_cols[col_idx]:
                    # Hallway selection (compact)
                    new_hallway_idx = st.selectbox(
                        f"H{grid_idx + 1}",
                        options=list(range(metadata['num_hallways'])),
                        index=grid_config['hallway_idx'],
                        key=f"hallway_{grid_idx}"
                    )
                    
                    if new_hallway_idx != grid_config['hallway_idx']:
                        st.session_state.grid_configs[grid_idx]['hallway_idx'] = new_hallway_idx
                        st.rerun()
                    
                    # Create and display observation grid (very small)
                    hallway_idx = grid_config['hallway_idx']
                    hallway_loc = tuple(metadata['hallway_locs'][hallway_idx])
                    agent_loc = grid_config['agent_loc']
                    goal_loc = tuple(metadata['goal_loc'])
                    
                    # Create observation visualization
                    grid_vis = create_observation_grid(
                        metadata['grid_size'], agent_loc, goal_loc, hallway_loc
                    )
                    
                    fig, ax = plt.subplots(figsize=(2, 2))  # Very small for combined view
                    ax.imshow(grid_vis)
                    ax.set_title(f'G{grid_idx + 1}H{hallway_idx}', fontsize=8)
                    ax.set_xticks(range(metadata['grid_size']))
                    ax.set_yticks(range(metadata['grid_size']))
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=6)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Agent location controls (very compact)
                    r_col, c_col = st.columns(2)
                    with r_col:
                        new_row = st.number_input(f"R", 
                                                min_value=0, max_value=metadata['grid_size']-1, 
                                                value=agent_loc[0], key=f"row_{grid_idx}")
                    with c_col:
                        new_col = st.number_input(f"C", 
                                                min_value=0, max_value=metadata['grid_size']-1, 
                                                value=agent_loc[1], key=f"col_{grid_idx}")
                    
                    if (new_row, new_col) != agent_loc:
                        if is_valid_location(new_row, new_col, metadata['grid_size'], goal_loc, hallway_loc):
                            st.session_state.grid_configs[grid_idx]['agent_loc'] = (new_row, new_col)
                            st.rerun()
                        else:
                            st.warning("Invalid!")
        
        st.markdown("---")
        st.write("**Combined Activation View**")
        
        # Create combined visualizations
        combined_figures = create_combined_visualization(
            combined_activation_data, 
            combined_activation_titles, 
            combined_grid_thumbnails
        )
        
        for activation_type, fig in combined_figures.items():
            if activation_type == 'Shared Representations' and show_shared_rep:
                st.pyplot(fig)
                plt.close(fig)
            elif activation_type.startswith('Task') and show_task_rep:
                task_idx = int(activation_type.split()[1])
                if task_idx in selected_tasks:
                    st.pyplot(fig)
                    plt.close(fig)
            elif activation_type == 'Q-Values' and show_q_values:
                st.pyplot(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()
