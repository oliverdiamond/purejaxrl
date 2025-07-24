# Neural Activation Visualizer

A Streamlit-based GUI for visualizing neural network activations in multi-task gridworld environments.

## Features

1. **Interactive Grid Visualization**: Display gridworld observations with agent (blue), goal (green), walls (black), and open spaces (white)
2. **Click-to-Move**: Click on any valid cell to move the agent to that location
3. **Multi-Grid Support**: Add/remove multiple grids and configure each independently
4. **Activation Visualization**: View shared representations, task-specific representations, and Q-values
5. **Flexible Display Modes**: Choose between separate displays (next to each grid) or stacked displays (combined view)
6. **Feature Normalization**: Automatic normalization with color-coding (blue=low, white=medium, red=high)
7. **Metadata Display**: View complete dataset and model configuration information

## Usage

### Option 1: Using the convenience script
```bash
./run_visualization.sh <filename>
```

### Option 2: Direct streamlit command
```bash
streamlit run purejaxrl/visualize_reps_easy_grid_with_task_specific_reps.py -- <filename>
```

### Example
```bash
./run_visualization.sh activations_N=5_n_tasks=3_shared_bottleneck=8_task_features=32_conv1=32_conv2=16_seed=0.pkl
```

## Interface Guide

### Control Panel
- **Activation Displays**: Choose which types of activations to visualize
  - Shared Representations: Common features across all tasks
  - Task Representations: Task-specific features (select which tasks to show)
  - Q-Values: Action values for the current state
- **Display Mode**: 
  - "Separate": Show activations next to each grid individually
  - "Stacked": Combine all activations in a single stacked view
- **Grid Management**: Add or remove observation grids

### Grid Interaction
- **Hallway Selection**: Choose which task/hallway configuration to display for each grid
- **Agent Movement**: 
  - Use the number inputs to manually set agent row/column
  - Agent can only be placed on valid locations (not walls or goal)
- **Visual Elements**:
  - Blue square: Agent location
  - Green square: Goal location
  - Black squares: Walls
  - White squares: Open spaces

### Activation Visualization
- **Feature Values**: Displayed as rectangles with color-coded squares
- **Color Scheme**: Blue (low) → White (medium) → Red (high)
- **Normalization**: 
  - Shared representations: Normalized across all observations
  - Task representations: Normalized per task across all observations
  - Q-values: Normalized per Q-value array

### Metadata Panel
- Expandable section showing environment parameters and model configuration
- Includes grid size, number of tasks, locations, and all hyperparameters

## Data Format

The visualization expects pickle files containing:
```python
{
    'activations': {
        hallway_idx: {
            'shared_rep': {(row, col): np.array},
            'task_rep': {(row, col): np.array},  # shape: (n_tasks, n_features)
            'q_values': {(row, col): np.array}   # shape: (n_tasks, n_actions)
        }
    },
    'metadata': {
        'grid_size': int,
        'num_hallways': int,
        'goal_loc': np.array,
        'start_loc': np.array,
        'hallway_locs': np.array,
        'config': dict  # Training configuration
    }
}
```

## Requirements

- streamlit >= 1.47.0
- matplotlib
- numpy
- pickle (standard library)

## Troubleshooting

1. **File not found**: Ensure the pickle file exists in `purejaxrl/data/`
2. **Invalid agent location**: Agent cannot be placed on walls or goal
3. **Missing activations**: Some agent locations may not have activation data
4. **Browser not opening**: Streamlit will show the local URL (usually http://localhost:8501)

## Notes

- The GUI initializes with one grid per hallway configuration
- Feature normalization ensures meaningful color comparisons
- All visualizations are interactive and update in real-time
- The interface is designed to be minimalist and responsive
