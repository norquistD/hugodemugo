import random
from learner import Learner
import configs

if configs.ENABLE_VISUALIZATION:
    import pygame
    pygame.init()


def game_loop():
    global display

    if configs.ENABLE_VISUALIZATION:
        display = pygame.display.set_mode((configs.DISPLAY_WIDTH, configs.DISPLAY_HEIGHT))
        pygame.display.set_caption("Snake")
        clock = pygame.time.Clock()

    # Starting position of snake
    snake_x = configs.DISPLAY_WIDTH / 2
    snake_y = configs.DISPLAY_HEIGHT / 2
    x_change = 0
    y_change = 0
    snake_list = [(snake_x, snake_y)]
    snake_length = 1
    # Track current direction as an int for the learner (0: left, 1: right, 2: up, 3: down)
    current_direction = 1  # start moving 'right' logically even if not yet moving

    # Create first food
    food_x = round(random.randrange(0, configs.DISPLAY_WIDTH - configs.BLOCK_SIZE) / 10.0) * 10.0
    food_y = round(random.randrange(0, configs.DISPLAY_HEIGHT - configs.BLOCK_SIZE) / 10.0) * 10.0

    dead = False
    reason = None
    while not dead:
        # Pump events to keep window responsive and allow quitting
        if configs.ENABLE_VISUALIZATION:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # Get action from agent
        action = learner.act(current_direction, snake_list, (food_x, food_y))
        if action == "left":
            x_change = -configs.BLOCK_SIZE
            y_change = 0
            current_direction = configs.DIRECTION_FROM_STRING["left"]
        elif action == "right":
            x_change = configs.BLOCK_SIZE
            y_change = 0
            current_direction = configs.DIRECTION_FROM_STRING["right"]
        elif action == "up":
            y_change = -configs.BLOCK_SIZE
            x_change = 0
            current_direction = configs.DIRECTION_FROM_STRING["up"]
        elif action == "down":
            y_change = configs.BLOCK_SIZE
            x_change = 0
            current_direction = configs.DIRECTION_FROM_STRING["down"]

        # Move snake
        snake_x += x_change
        snake_y += y_change
        snake_head = (snake_x, snake_y)
        snake_list.append(snake_head)

        # Check if snake is off screen
        if snake_x >= configs.DISPLAY_WIDTH or snake_x < 0 or snake_y >= configs.DISPLAY_HEIGHT or snake_y < 0:
            reason = "Screen"
            dead = True

        # Check if snake hit tail
        if snake_head in snake_list[:-1]:
            reason = "Tail"
            dead = True

        # Check if snake ate food
        if snake_x == food_x and snake_y == food_y:
            food_x = round(random.randrange(0, configs.DISPLAY_WIDTH - configs.BLOCK_SIZE) / 10.0) * 10.0
            food_y = round(random.randrange(0, configs.DISPLAY_HEIGHT - configs.BLOCK_SIZE) / 10.0) * 10.0
            snake_length += 1

        # Delete the last cell since we just added a head for moving, unless we ate a food
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Draw food, snake and update score (only if visualization is enabled)
        if configs.ENABLE_VISUALIZATION:
            display.fill(configs.BLUE)
            draw_food(food_x, food_y)
            draw_snake(snake_list)
            draw_score(snake_length - 1)
            pygame.display.update()

        # Update Q Table
        learner.update_q_values(reason)

        # Next Frame (only if visualization is enabled)
        if configs.ENABLE_VISUALIZATION:
            clock.tick(configs.FRAME_SPEED)

    return snake_length - 1, reason


def draw_food(food_x, food_y):
    if configs.ENABLE_VISUALIZATION:
        pygame.draw.rect(display, configs.GREEN, [food_x, food_y, configs.BLOCK_SIZE, configs.BLOCK_SIZE])


def draw_score(score):
    if configs.ENABLE_VISUALIZATION:
        font = pygame.font.SysFont("comicsansms", 35)
        value = font.render(f"Score: {score}", True, configs.YELLOW)
        display.blit(value, [0, 0])


def draw_snake(snake_list):
    if configs.ENABLE_VISUALIZATION:
        for x in snake_list:
            pygame.draw.rect(display, configs.BLACK, [x[0], x[1], configs.BLOCK_SIZE, configs.BLOCK_SIZE])


game_count = 1

learner = Learner(configs.DISPLAY_WIDTH, configs.DISPLAY_HEIGHT, configs.BLOCK_SIZE)

high_score = 0
while True:
    learner.reset()
    learner.epsilon *= learner.epsilon_decay
    score, reason = game_loop()
    print(
        f"Games: {game_count}; Score: {score}; Reason: {reason}; Current highscore: {high_score}"
    )  # Output results of each game to console to monitor as agent is training
    if score > high_score:
        high_score = score
        learner.save_qvalues(path="highscores/highscore" + str(high_score) + ".json")
    game_count += 1
    if game_count % configs.QVALUES_SAVE_INTERVAL == 0:  # Save qvalues every qvalue_dump_n games
        print("Save Qvals")
        learner.save_qvalues()
