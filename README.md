# Snake Q-Learning

## Project Overview

This project implements a classic Snake game with an agent that learns to play using Q-learning, a reinforcement learning algorithm. The agent explores the game environment, learns optimal actions through trial and error, and improves its performance over time by updating a Q-table that maps game states to action values.

## Contributors

- *Dylan Norquist*
- *Hugo Garrido*

## Project Structure

### Core Files

- **`agent.py`**: Implements the Q-learning agent with epsilon-greedy action selection and Q-value updates using the Bellman equation.
- **`game.py`**: Main entry point that orchestrates the training loop, manages progress tracking, and coordinates between the agent, game engine, and renderer.
- **`snake_game.py`**: Pure game engine that encapsulates Snake game logic, state management, and collision detection without any visualization or learning components.
- **`renderer.py`**: Handles all pygame-based visualization, including drawing the snake, food, and updating the window title with the current score.
- **`state_representations.py`**: Defines different state representation strategies (basic and naive) that encode the game state for the Q-learning algorithm.
- **`configs.py`**: Centralized configuration management using Pydantic, implementing a singleton pattern for global settings access and CLI argument parsing with tyro.

### Supporting Files

- **`train.ps1`**: PowerShell script for batch training and evaluation across different state representations.
- **`results.txt`**: Contains training and evaluation results including scores, episode counts, and performance metrics.
- **`pyproject.toml`**: Python project configuration file specifying dependencies and Python version requirements.
- **`requirements.txt`**: Duplicate Python project configuration file for dependencies

## Installation

### Prerequisites

- **Python 3.12 or higher**

### Dependencies

The project requires the following Python packages:

- `pygame>=2.6.1` - Game visualization and window management
- `tqdm>=4.66.0` - Progress bar display
- `pydantic>=2.0.0` - Configuration validation and settings management
- `pydantic-settings>=2.0.0` - Settings management for Pydantic
- `tyro>=0.8.0` - Command-line argument parsing

### Installation Methods

#### Using Setup script (default):

```bash
./install.ps1
```
or 
```bash
pip install -r requirements.txt
```

#### Using `uv` (Recommended)

If you have `uv` installed:

```bash
uv sync
```

This will automatically install all dependencies specified in `pyproject.toml`.

#### Using `pip`

```bash
pip install pygame tqdm pydantic pydantic-settings tyro
```

Or install from the project file:

```bash
pip install -e .
```

## Running the Code

### Basic Training

Run training with default settings (no visualization):

```bash
python game.py
```

Or using `uv`:

```bash
uv run game.py
```

### Command-Line Arguments

The project supports several command-line arguments:

- **`--steps <number>`**: Maximum number of total steps for training or evaluation (default: unlimited)
  - Example: `--steps 1000000` limits training to 1 million steps
  - Works for both training and evaluation modes

- **`--state <representation>`**: Choose the state representation strategy
  - Options: `basic`, or `naive`
  - Default: `basic`
  - Example: `--state basic` uses the basic state representation

- **`--visuals`**: Enable pygame visualization (opens a game window)
  - Default: disabled (headless training)
  - Example: `--visuals` enables the game window

- **`--evals`**: Run in evaluation mode (no training, epsilon=0, no Q-table updates)
  - Default: training mode
  - Example: `--evals` runs evaluation mode

### Example Commands

**Train with visualization using basic state representation:**

```bash
uv run game.py --state basic --visuals
```

**Train for 10 million steps with basic state representation:**

```bash
uv run game.py --steps 10000000 --state basic
```

**Evaluate a trained agent (no learning):**

```bash
uv run game.py --evals --steps 1000000 --state basic
```

**Train naive state representation (demonstration):**

```bash
uv run game.py --state naive --steps 10000000
```

### Batch Training

Use the provided PowerShell script for batch training:

```powershell
.\train.ps1
```

This script trains and evaluates all three state representations sequentially.

## Results and Outputs

### Q-Values Storage

Q-tables (Q-values) are stored as JSON files in state-specific directories:

- **Location**: `{state}/qvalues.json` (e.g., `basic/qvalues.json`, `naive/qvalues.json`)
- **Format**: JSON dictionary mapping state strings to arrays of Q-values for each action
- **Persistence**: Q-values are automatically saved:
  - Periodically during training (every `QVALUES_SAVE_INTERVAL` games, default: 1000)
  - When a new high score is achieved
  - On program termination (Ctrl+C or normal exit)

### Highscore Snapshots

When the agent achieves a new high score, a snapshot of the Q-table is saved:

- **Location**: `{state}/highscores/highscore{N}.json` (e.g., `basic/highscores/highscore50.json`)
- **Naming**: Files are named with the score value (e.g., `highscore50.json` for score 50)
- **Purpose**: Preserve Q-tables at milestone scores for analysis or rollback

### Results File

The `results.txt` file contains training and evaluation output:

- **Content**: Console output from training/evaluation runs including:
  - Final step counts
  - Training progress with scores, reasons, epsilon values
  - Evaluation statistics (episodes, average score, max/min scores, total steps)
- **Interpretation**: 
  - Higher scores indicate better performance
  - "Reason" field shows termination cause: "Tail" (hit tail), "Screen" (hit wall), "Steps" (max steps exceeded)
  - Epsilon values show exploration rate (decreases over time)
  - Average reward indicates learning progress

### Progress Bar Metrics

During training, the progress bar displays:

- **Score**: Current game score (snake length - 1)
- **Reason**: Game termination reason ("Tail", "Screen", or "Steps")
- **High**: Highest score achieved so far
- **Epsilon**: Current exploration rate (0.0 to 1.0)
- **AvgReward**: Average reward over last 100 games
- **Games**: Number of games played (shown when `--steps` is set)

During evaluation, the progress bar displays:

- **Score**: Current game score
- **Avg**: Average score across all evaluation episodes
- **Max**: Maximum score achieved
- **Reason**: Game termination reason

### Interpreting Results

- **Training Success**: Look for increasing high scores and average rewards over time
- **State Representation Comparison**: Compare final scores between `basic` and `naive` representations
- **Evaluation Performance**: Use `--evals` mode to test agent performance without further learning (epsilon=0)
- **Learning Progress**: Monitor epsilon decay and average reward trends in the progress bar

## Configuration

Most settings can be configured in `configs.py` or via environment variables (with `SNAKE_` prefix):

- **Learning Parameters**: `EPSILON`, `LEARNING_RATE`, `DISCOUNT`
- **Game Settings**: `BLOCK_SIZE`, `DISPLAY_WIDTH`, `DISPLAY_HEIGHT`, `MAX_STEPS_SINCE_LAST_FRUIT`
- **Training Settings**: `QVALUES_SAVE_INTERVAL`, `MAX_STEPS`, `ENABLE_VISUALIZATION`

For more details, see the docstrings in `configs.py` or `pyproject.toml`.
