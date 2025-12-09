"""
Q-Learning Agent Implementation

Implements the Q-learning agent that learns to play Snake. Used by game.py for action
selection and Q-value updates. Reads configuration from configs.py and uses state
representation functions from state_representations.py to encode game states. Saves
Q-tables to JSON files in state-specific directories (basic/, naive/).
"""

import random
import json
from typing import Optional, Dict, List
from configs import get_config
import os
from state_representations import STATE_REPRESENTATIONS, GameState


class Agent:
    def __init__(self, display_width: int, display_height: int, block_size: int) -> None:
        # Game parameters
        self.display_width = display_width
        self.display_height = display_height
        self.block_size = block_size

        # Cache config for later use
        self.cfg = get_config()

        # Learning parameters
        self.epsilon = self.cfg.EPSILON
        self.learning_rate = self.cfg.LEARNING_RATE
        self.discount = self.cfg.DISCOUNT

        # State representation strategy
        self.state_strategy = self.cfg.STATE_REPRESENTATION
        if self.state_strategy not in STATE_REPRESENTATIONS:
            raise ValueError(
                f"Unknown state representation: {self.state_strategy}. "
                f"Available options: {list(STATE_REPRESENTATIONS.keys())}"
            )
        self.get_state_func = STATE_REPRESENTATIONS[self.state_strategy]

        # State/Action history
        self.qvalues = self.load_qvalues()
        self.history = []

        # Action space
        self.actions = self.cfg.ACTIONS

    def reset(self) -> None:
        self.history = []

    def load_qvalues(self, path: Optional[str] = None) -> Dict[str, List[float]]:
        if path is None:
            path = self.cfg.DEFAULT_QVALUES_PATH
        if not os.path.exists(path):
            print("No qvalues file found, creating new one")
            return {}

        with open(path, "r") as f:
            qvalues = json.load(f)
        print("Qvalues loaded from", path)
        return qvalues

    def save_qvalues(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self.cfg.DEFAULT_QVALUES_PATH
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path:  # Only create directory if path contains a directory component
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.qvalues, f)

    def act(self, direction: int, snake: List[tuple[float, float]], food: tuple[float, float]) -> str:
        state = self._get_state(direction, snake, food)
        state_str = self._get_state_str(state)

        # Initialize Q-values for new states
        if state_str not in self.qvalues:
            self.qvalues[state_str] = [0.0] * len(self.actions)

        # Epsilon greedy
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            action_key = random.choices(list(self.actions.keys()))[0]
        else:
            state_scores = self.qvalues[state_str]
            action_key = state_scores.index(max(state_scores))
        action_val = self.actions[action_key]

        # Remember the actions it took at each state
        self.history.append({"state": state, "action": action_key})
        return action_val

    def update(self, reason: Optional[str]) -> float:
        """
        Update Q-values based on episode history (Q-learning convention).
        
        Args:
            reason: Termination reason ("Screen", "Tail", "Steps", or None)
            
        Returns:
            Total reward for the episode
        """
        # Process history once at the end of the game
        if not self.history:
            return 0.0

        history = self.history
        history_len = len(history)
        total_reward = 0.0

        # Handle terminal state (snake died) - apply negative reward to last state
        if reason:
            if history_len > 0:
                terminal_state = history[-1]["state"]
                terminal_action = history[-1]["action"]
                state_str = self._get_state_str(terminal_state)

                # Initialize Q-values for new states
                if state_str not in self.qvalues:
                    self.qvalues[state_str] = [0.0] * len(self.actions)

                if reason == "Steps":  # Snake took too many steps
                    reward = -3
                else:
                    reward = -30
                total_reward += reward
                self.qvalues[state_str][terminal_action] = (
                    1 - self.learning_rate
                ) * self.qvalues[state_str][
                    terminal_action
                ] + self.learning_rate * reward  # Bellman equation - there is no future state since game is over

        # Process all state transitions in forward order (more efficient than reverse)
        for i in range(history_len - 1):
            s0 = history[i]["state"]  # current state
            a0 = history[i]["action"]  # action taken at current state
            s1 = history[i + 1]["state"]  # next state

            x1 = s0.distance[0]  # x distance at current state
            y1 = s0.distance[1]  # y distance at current state

            x2 = s1.distance[0]  # x distance at next state
            y2 = s1.distance[1]  # y distance at next state

            if s0.food != s1.food:  # Snake ate a food, positive reward
                reward = 10
            elif abs(x1) > abs(x2) or abs(y1) > abs(
                y2
            ):  # Snake is closer to the food, positive reward
                reward = 0.001
            else:
                reward = -0.001  # Snake is further from the food, negative reward

            total_reward += reward

            state_str = self._get_state_str(s0)
            new_state_str = self._get_state_str(s1)

            # Initialize Q-values for new states
            if state_str not in self.qvalues:
                self.qvalues[state_str] = [0.0] * len(self.actions)
            if new_state_str not in self.qvalues:
                self.qvalues[new_state_str] = [0.0] * len(self.actions)

            self.qvalues[state_str][a0] = (1 - self.learning_rate) * (
                self.qvalues[state_str][a0]
            ) + self.learning_rate * (
                reward + self.discount * max(self.qvalues[new_state_str])
            )  # Bellman equation

        return total_reward

    def _get_state(self, direction: int, snake: List[tuple[float, float]], food: tuple[float, float]) -> GameState:
        # Call the state representation function based on config
        return self.get_state_func(
            direction,
            snake,
            food,
            self.display_width,
            self.display_height,
            self.block_size,
        )

    def _get_state_str(self, state: GameState) -> str:
        # Return cached state string if available, otherwise compute and cache it
        # Include strategy name to avoid Q-value collisions between different strategies
        current_strategy = self.cfg.STATE_REPRESENTATION
        strategy_name = getattr(state, "strategy_name", current_strategy)

        # Check if we have a cached full state string (with strategy prefix)
        if hasattr(state, "_full_state_str") and state._full_state_str is not None:
            return state._full_state_str

        # Compute base state string
        if hasattr(state, "_state_str") and state._state_str is not None:
            base_state_str = state._state_str
        else:
            # Include all position elements to support basic representation
            base_state_str = str((*state.position, state.surroundings, state.direction))
            state._state_str = base_state_str

        # Create full state string with strategy prefix and cache it
        full_state_str = f"{strategy_name}:{base_state_str}"
        state._full_state_str = full_state_str
        return full_state_str
