# Snake Q Learning

## What is this?

This project is a classic Snake game with an automated agent that learns to play using Q-learning, a type of reinforcement learning. The code uses `pygame` for graphics and learning logic is implemented in the `Learner` class.

## How it works

- Each game, the AI (the "learner") controls the snake.
- After every action, the learner updates its Q-table (knowledge of states and actions) based on reward (e.g., eating food, hitting the wall or tail).
- Over many games, the AI improves by exploring and then exploiting the best moves it has learned.
- The Q-table is periodically saved to disk so that learning can continue from the last state.

## How to run

1. **Requirements**  
   - Python 3.x  
   - `pygame`  
     ```
     pip install pygame
     ```

2. **Start the Game**  
   Run the main file:
   ```
   python game.py
   ```
   A window will open showing the snake game. The AI will play automatically.

3. **Monitor Learning**  
   - Console output will display scores, reasons for game over, and high scores as training progresses.
   - High scores and Q-tables are saved to disk in `highscores/` and as JSON files.

4. **Stopping and Continuing Training**  
   - Press the close button on the game window (QUIT) to stop.
   - Next run will continue learning using the previously saved Q-table if available.

## Files

- `game.py`: Main script to run the game and the learning loop.
- `learner.py`: Contains the Q-learning logic and data structure.
- `qTable/`: Directory for storing Q-tables.
- `highscores/`: Directory for saving highest scoring Q-tables.
- `LICENSE`: MIT License for open-source use.

## More Info

You can modify learning rates, Q-table interval saves, or snake/block/screen settings at the top of `game.py` and `learner.py` to experiment with the learning process.

For deeper understanding or to customize, inspect the `Learner` class in `learner.py`.

