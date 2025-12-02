# Configuration constants for Snake Q-Learning game

# Colors (RGB tuples)
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

# Display settings
BLOCK_SIZE = 10
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 400

# Game settings
QVALUES_SAVE_INTERVAL = 1000
FRAME_SPEED = 500000
ENABLE_VISUALIZATION = False

# Learning parameters
EPSILON = 0.1
EPSILON_DECAY = 0.999
LEARNING_RATE = 0.03
DISCOUNT = 0.9

# Action mappings
ACTIONS = {0: "left", 1: "right", 2: "up", 3: "down"}
DIRECTION_FROM_STRING = {"left": 0, "right": 1, "up": 2, "down": 3}

# File paths
DEFAULT_QVALUES_PATH = "qvalues.json"
HIGHSCORES_DIR = "highscores"

