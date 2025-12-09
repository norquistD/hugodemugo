"""
Pygame-Based Renderer for Visualization

Handles all pygame visualization. Used by game.py to render game state from
snake_game.py. Reads visualization settings from configs.py. Completely optional
and can be disabled for headless training. Receives game state (snake, food, score)
and renders it to the pygame window.
"""

import pygame
from typing import Dict, Any
from configs import get_config


class Renderer:
    """Handles all pygame rendering for the Snake game."""

    def __init__(self):
        """Initialize pygame and create display."""
        self.cfg = get_config()
        if self.cfg.ENABLE_VISUALIZATION:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.cfg.DISPLAY_WIDTH, self.cfg.DISPLAY_HEIGHT)
            )
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None

    def handle_events(self):
        """Handle pygame events (quit, etc.). Returns False if should quit."""
        if not self.cfg.ENABLE_VISUALIZATION:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True

    def render(self, game_state: Dict[str, Any]) -> None:
        """Render the current game state."""
        if not self.cfg.ENABLE_VISUALIZATION:
            return

        # Calculate score and update window title
        score = game_state["length"] - 1
        pygame.display.set_caption(f"Snake - Score: {score}")

        self.display.fill(self.cfg.BLUE)
        self._draw_food(game_state["food"])
        self._draw_snake(game_state["snake"])
        pygame.display.update()

        # Check if 'c' key is held to slow down to 60 fps
        keys = pygame.key.get_pressed()
        if keys[pygame.K_c]:
            self.clock.tick(60)
        else:
            self.clock.tick(self.cfg.FRAME_SPEED)

    def _draw_food(self, food: tuple[float, float]) -> None:
        """Draw food on the screen."""
        pygame.draw.rect(
            self.display,
            self.cfg.GREEN,
            [food[0], food[1], self.cfg.BLOCK_SIZE, self.cfg.BLOCK_SIZE],
        )

    def _draw_snake(self, snake_list: list[tuple[float, float]]) -> None:
        """Draw snake on the screen."""
        for segment in snake_list:
            pygame.draw.rect(
                self.display,
                self.cfg.BLACK,
                [segment[0], segment[1], self.cfg.BLOCK_SIZE, self.cfg.BLOCK_SIZE],
            )

    def _draw_score(self, score: int) -> None:
        """Draw score on the screen."""
        font = pygame.font.SysFont("comicsansms", 35)
        value = font.render(f"Score: {score}", True, self.cfg.YELLOW)
        self.display.blit(value, [0, 0])

    def close(self) -> None:
        """Close/cleanup the renderer (Gym convention)."""
        if self.cfg.ENABLE_VISUALIZATION:
            pygame.quit()
