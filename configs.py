"""
Configuration Management

Centralized configuration using Pydantic and tyro for CLI parsing. Implements
singleton pattern accessed via get_config() by all modules (agent.py, game.py,
snake_game.py, renderer.py). Parses command-line arguments (--steps, --state,
--visuals, --evals) and provides type-safe settings for game parameters, learning
hyperparameters, and state representation selection.
"""

import sys
from typing import Optional, Literal, List
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import tyro


class SnakeConfig(BaseSettings):
    """Configuration settings for the Snake Q-Learning game."""

    model_config = SettingsConfigDict(
        env_prefix="SNAKE_",
        case_sensitive=False,
        extra="ignore",
    )

    # Colors (RGB tuples) - These are constants, not settings
    YELLOW: tuple[int, int, int] = (255, 255, 102)
    BLACK: tuple[int, int, int] = (0, 0, 0)
    GREEN: tuple[int, int, int] = (0, 255, 0)
    BLUE: tuple[int, int, int] = (50, 153, 213)

    # Display settings
    BLOCK_SIZE: int = 100
    DISPLAY_WIDTH: int = 1000
    DISPLAY_HEIGHT: int = 1000

    # Game settings
    QVALUES_SAVE_INTERVAL: int = Field(default=1_000, ge=1)
    FRAME_SPEED: int = Field(default=30, ge=1)
    ENABLE_VISUALIZATION: bool = False
    EVALS_MODE: bool = False
    # Maximum number of total steps (None = unlimited, works for both training and evaluation)
    MAX_STEPS: Optional[int] = Field(default=None, ge=1)
    # Maximum number of steps since last fruit eaten (game ends if exceeded)
    MAX_STEPS_SINCE_LAST_FRUIT: int = Field(default=2_000, ge=1)

    # Learning parameters
    EPSILON: float = Field(default=0.02, ge=0.0, le=1.0)
    LEARNING_RATE: float = Field(default=0.03, ge=0.0, le=1.0)
    DISCOUNT: float = Field(default=0.9, ge=0.0, le=1.0)

    # State representation strategy
    STATE_REPRESENTATION: Literal["basic", "naive"] = Field(
        default="basic", pattern="^(basic|naive)$"
    )

    # Action mappings (constants)
    ACTIONS: dict[int, str] = {0: "left", 1: "right", 2: "up", 3: "down"}
    DIRECTION_FROM_STRING: dict[str, int] = {
        "left": 0,
        "right": 1,
        "up": 2,
        "down": 3,
    }

    # File paths (computed properties)
    @computed_field
    @property
    def DEFAULT_QVALUES_PATH(self) -> str:
        """Default path for Q-values file based on state representation."""
        return f"{self.STATE_REPRESENTATION}/qvalues.json"

    @computed_field
    @property
    def HIGHSCORES_DIR(self) -> str:
        """Directory for highscore files based on state representation."""
        return f"{self.STATE_REPRESENTATION}/highscores"


def _init_config_from_cli(args: Optional[List[str]] = None) -> SnakeConfig:
    """
    Initialize configuration from command-line arguments using tyro.

    Tyro automatically generates CLI arguments from the dataclass fields.
    Only fields that should be configurable via CLI are exposed.
    """
    from dataclasses import dataclass, field

    @dataclass
    class CLIConfig:
        """CLI configuration options for Snake Q-Learning Training."""

        steps: Optional[int] = field(
            default=None,
            metadata={
                "help": "Maximum number of total steps (None = unlimited, works for both training and evaluation)"
            },
        )
        visuals: bool = field(
            default=False, metadata={"help": "Enable pygame visualization"}
        )
        state: Literal["basic", "naive"] = field(
            default="basic",
            metadata={"help": "State representation (basic or naive)"},
        )
        evals: bool = field(
            default=False,
            metadata={"help": "Run evaluation mode (no training, epsilon=0, no Q-table updates)"},
        )

    # Parse CLI arguments using tyro
    # If args is None, tyro will parse from sys.argv
    cli_args = tyro.cli(CLIConfig, args=args)

    # Convert CLI args to config dict, only including non-None/non-default values
    config_dict = {}
    if cli_args.steps is not None:
        config_dict["MAX_STEPS"] = cli_args.steps
    if cli_args.visuals:
        config_dict["ENABLE_VISUALIZATION"] = True
    if cli_args.state is not None:
        config_dict["STATE_REPRESENTATION"] = cli_args.state
    if cli_args.evals:
        config_dict["EVALS_MODE"] = True

    # Create config with defaults (BaseSettings will read from environment variables)
    # Then update with CLI args if any were provided
    base_config = SnakeConfig()
    if config_dict:
        return base_config.model_copy(update=config_dict)
    return base_config


# Private module-level variable to hold the singleton instance
_configs: SnakeConfig | None = None


def init_config(args: Optional[List[str]] = None) -> SnakeConfig:
    """
    Initialize and return the global config instance from CLI arguments.
    This should be called once at application startup.
    """
    global _configs
    _configs = _init_config_from_cli(args)
    return _configs


def get_config() -> SnakeConfig:
    """
    Get the current config singleton instance.
    If not initialized, creates a default instance.
    """
    global _configs
    if _configs is None:
        _configs = SnakeConfig()
    return _configs
