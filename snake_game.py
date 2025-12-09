"""
Pure Snake Game Engine

Implements core Snake game logic (movement, collision detection, food generation)
without visualization or learning. Used by game.py to run game episodes. Provides
Gym-style step() interface returning (observation, reward, done, info). Game state
is passed to agent.py for action selection and to renderer.py for visualization.
Reads game settings from configs.py.
"""

import random
from typing import Optional, Tuple, Dict, Any
from configs import get_config


class SnakeGame:
    """Snake game engine with step-based interface."""

    def __init__(self):
        """Initialize a new game."""
        self.cfg = get_config()
        self.reset()

    def reset(self) -> None:
        """Reset the game to initial state."""
        # Starting position of snake
        self.snake_x = self.cfg.DISPLAY_WIDTH / 2
        self.snake_y = self.cfg.DISPLAY_HEIGHT / 2
        self.snake_list = [(self.snake_x, self.snake_y)]
        # Use set for O(1) collision detection
        self.snake_set = {self.snake_list[0]}
        self.snake_length = 1
        # Track current direction (0: left, 1: right, 2: up, 3: down)
        self.current_direction = 1  # start moving 'right'

        # Create first food
        self.food_x, self.food_y = self._generate_food_position()

        self.dead = False
        self.reason = None
        self.steps = 0
        self.steps_since_last_fruit = 0

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one game step with the given action (Gym-style interface).

        Args:
            action: Action string ("left", "right", "up", "down")

        Returns:
            Tuple of (observation, reward, done, info) following Gym convention
            - observation: Current game state/observation
            - reward: Reward for this step (1.0 for eating food, 0.0 otherwise, negative for death)
            - done: True if game is over
            - info: Dictionary with additional info (reason, score, etc.)
        """
        if self.dead:
            return self.get_observation(), -1.0, True, {"reason": self.reason}

        self.steps += 1
        self.steps_since_last_fruit += 1

        # Check if max steps since last fruit exceeded
        if self.steps_since_last_fruit >= self.cfg.MAX_STEPS_SINCE_LAST_FRUIT:
            self.dead = True
            self.reason = "Steps"
            return self.get_observation(), -1.0, True, {"reason": self.reason}

        # Calculate movement
        dx, dy, new_direction = self._action_to_movement(action)
        new_x = self.snake_x + dx
        new_y = self.snake_y + dy
        new_head = (new_x, new_y)

        # Check collision
        collision, reason = self._check_collision(new_x, new_y, self.snake_set)
        if collision:
            self.dead = True
            self.reason = reason
            return self.get_observation(), -1.0, True, {"reason": reason}

        # Move snake
        self.snake_x = new_x
        self.snake_y = new_y
        self.current_direction = new_direction
        self.snake_list.append(new_head)
        self.snake_set.add(new_head)

        # Check if snake ate food
        reward = 0.0
        if self.snake_x == self.food_x and self.snake_y == self.food_y:
            self.food_x, self.food_y = self._generate_food_position()
            self.snake_length += 1
            self.steps_since_last_fruit = 0  # Reset counter when fruit is eaten
            reward = 1.0
        else:
            # Remove tail if we didn't eat food
            if len(self.snake_list) > self.snake_length:
                tail_to_remove = self.snake_list[0]
                self.snake_set.discard(tail_to_remove)
                del self.snake_list[0]

        info = {"score": self.get_score(), "steps": self.steps}
        return self.get_observation(), reward, False, info

    def get_observation(self) -> Dict[str, Any]:
        """
        Get current game observation for agent (Gym convention).

        Returns:
            Dictionary with snake, food, direction, and other state info
        """
        return {
            "snake": self.snake_list,
            "food": (self.food_x, self.food_y),
            "direction": self.current_direction,
            "length": self.snake_length,
            "steps": self.steps,
        }

    @property
    def done(self) -> bool:
        """Check if game is over (Gym convention)."""
        return self.dead

    def get_score(self) -> int:
        """Get current score (snake length - 1)."""
        return self.snake_length - 1

    def _action_to_movement(self, action: str) -> Tuple[int, int, int]:
        """Convert action string to (dx, dy, direction) tuple."""
        action_map = {
            "left": (
                -self.cfg.BLOCK_SIZE,
                0,
                self.cfg.DIRECTION_FROM_STRING["left"],
            ),
            "right": (
                self.cfg.BLOCK_SIZE,
                0,
                self.cfg.DIRECTION_FROM_STRING["right"],
            ),
            "up": (
                0,
                -self.cfg.BLOCK_SIZE,
                self.cfg.DIRECTION_FROM_STRING["up"],
            ),
            "down": (
                0,
                self.cfg.BLOCK_SIZE,
                self.cfg.DIRECTION_FROM_STRING["down"],
            ),
        }
        return action_map[action]

    def _check_collision(
        self, x: float, y: float, snake_set: set[tuple[float, float]]
    ) -> Tuple[bool, Optional[str]]:
        """Check if position (x, y) collides with walls or snake body."""
        if (
            x >= self.cfg.DISPLAY_WIDTH
            or x < 0
            or y >= self.cfg.DISPLAY_HEIGHT
            or y < 0
        ):
            return True, "Screen"
        if (x, y) in snake_set:
            return True, "Tail"
        return False, None

    def _generate_food_position(self) -> Tuple[float, float]:
        """Generate a random food position aligned to the grid."""
        food_x = (
            round(
                random.randrange(0, self.cfg.DISPLAY_WIDTH - self.cfg.BLOCK_SIZE)
                / 10.0
            )
            * 10.0
        )
        food_y = (
            round(
                random.randrange(0, self.cfg.DISPLAY_HEIGHT - self.cfg.BLOCK_SIZE)
                / 10.0
            )
            * 10.0
        )
        return food_x, food_y

    def close(self) -> None:
        """Close/cleanup the environment (Gym convention)."""
        # This is kept incase I needed to add cleanup logic in the future
        pass

