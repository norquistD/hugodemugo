"""
State Representation Functions for Q-Learning

Defines state encoding strategies (basic and naive) that convert game state
from snake_game.py into GameState objects for agent.py. Used by agent.py via the
STATE_REPRESENTATIONS dictionary. Strategy selection is configured in configs.py.
Each strategy produces different state encodings, affecting learning performance.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class GameState:
    distance: Tuple[int, int]
    position: Tuple[str, ...]
    surroundings: str
    food: Tuple[float, float]
    direction: int
    _state_str: Optional[str] = field(
        default=None, init=False
    )  # Cached state string to avoid repeated conversions | Cached for performance
    strategy_name: str = field(
        default="basic", init=False
    )  # Strategy name for state string | Used for state string caching

    def __post_init__(self) -> None:
        # Cache the state string on creation
        if self._state_str is None:
            # Include all position elements to support basic representation
            self._state_str = str(
                (*self.position, self.surroundings, self.direction)
            )


def _get_food_position(
    snake_head: Tuple[float, float], food: Tuple[float, float]
) -> Tuple[int, int, str, str]:
    """
    Calculate food position relative to snake head.

    Returns:
        Tuple of (dist_x, dist_y, pos_x, pos_y)
        - dist_x, dist_y: Raw distance values
        - pos_x, pos_y: Categorical position ("0", "1", "NA")
    """
    dist_x = food[0] - snake_head[0]
    dist_y = food[1] - snake_head[1]

    if dist_x > 0:
        pos_x = "1"  # Food is to the right of the snake
    elif dist_x < 0:
        pos_x = "0"  # Food is to the left of the snake
    else:
        pos_x = "NA"  # Food and snake are on the same X coordinate

    if dist_y > 0:
        pos_y = "3"  # Food is below snake
    elif dist_y < 0:
        pos_y = "2"  # Food is above snake
    else:
        pos_y = "NA"  # Food and snake are on the same Y coordinate

    return dist_x, dist_y, pos_x, pos_y


def _get_surroundings(
    snake_head: Tuple[float, float],
    snake_body_set: set[Tuple[float, float]],
    display_width: int,
    display_height: int,
    block_size: int,
) -> str:
    """
    Calculate surroundings (danger/safe) in 4 directions around snake head.

    Returns:
        String of 4 characters ("0" = safe, "1" = danger) for [left, right, up, down]
    """
    adjacent_squares = [
        (snake_head[0] - block_size, snake_head[1]),  # left
        (snake_head[0] + block_size, snake_head[1]),  # right
        (snake_head[0], snake_head[1] - block_size),  # up
        (snake_head[0], snake_head[1] + block_size),  # down
    ]

    surrounding_list = []
    for square in adjacent_squares:
        if square[0] < 0 or square[1] < 0:  # off screen left or top
            surrounding_list.append("1")
        elif (
            square[0] >= display_width or square[1] >= display_height
        ):  # off screen right or bottom
            surrounding_list.append("1")
        elif square in snake_body_set:  # part of tail - O(1) lookup
            surrounding_list.append("1")
        else:
            surrounding_list.append("0")
    return "".join(surrounding_list)


def get_state_basic(
    direction: int,
    snake: List[Tuple[float, float]],
    food: Tuple[float, float],
    display_width: int,
    display_height: int,
    block_size: int,
) -> GameState:
    """
    Basic state representation:
    - Food position relative to snake (left/right, above/below)
    - Surroundings (4 directions: danger or safe)
    - Current direction
    """
    snake_head = snake[-1]
    snake_body_set = set(snake[:-1]) if len(snake) > 1 else set()

    dist_x, dist_y, pos_x, pos_y = _get_food_position(snake_head, food)
    surroundings = _get_surroundings(
        snake_head, snake_body_set, display_width, display_height, block_size
    )

    state = GameState(
        distance=(dist_x, dist_y),
        position=(pos_x, pos_y),
        surroundings=surroundings,
        food=food,
        direction=direction,
    )
    state.strategy_name = "basic"
    return state


def get_state_naive(
    direction: int,
    snake: List[Tuple[float, float]],
    food: Tuple[float, float],
    display_width: int,
    display_height: int,
    block_size: int,
) -> GameState:
    """
    Naive state representation with minimal/no processing:
    - Only uses direction and raw food coordinates
    - No relative food positioning (loses spatial relationship)
    - No surroundings information (cannot avoid collisions)
    - Demonstrates why proper state representation is critical
    
    This naive approach will perform poorly because:
    - Raw coordinates create a huge state space (every position is unique)
    - No information about dangers (walls, tail)
    - Cannot learn spatial relationships effectively
    """
    raw_food_x = str(int(food[0]))
    raw_food_y = str(int(food[1]))

    snake_head = snake[-1]
    # Use raw coordinates for distance (naive - no relative positioning)
    # Must be numeric for agent.update() which uses abs() on these values
    snake_x = int(snake_head[0])
    snake_y = int(snake_head[1])
    
    state = GameState(
        distance=(snake_x, snake_y),  # Raw snake position (not relative to food)
        position=(raw_food_x, raw_food_y),  # Raw food coordinates as strings
        surroundings="NA",  # No danger detection
        food=food,
        direction=direction,
    )
    state.strategy_name = "naive"
    return state


# Dictionary mapping strategy names to their functions
STATE_REPRESENTATIONS = {
    "basic": get_state_basic,
    "naive": get_state_naive,
}
