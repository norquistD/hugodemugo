import random
import json
import dataclasses
import configs


@dataclasses.dataclass
class GameState:
    distance: tuple
    position: tuple
    surroundings: str
    food: tuple
    direction: int


class Learner(object):
    def __init__(self, display_width, display_height, block_size):
        # Game parameters
        self.display_width = display_width
        self.display_height = display_height
        self.block_size = block_size

        # Learning parameters
        self.epsilon = configs.EPSILON
        self.epsilon_decay = configs.EPSILON_DECAY
        self.learning_rate = configs.LEARNING_RATE
        self.discount = configs.DISCOUNT

        # State/Action history
        self.qvalues = self.load_qvalues()
        self.history = []

        # Action space
        self.actions = configs.ACTIONS

    def reset(self):
        self.history = []

    def load_qvalues(self, path=configs.DEFAULT_QVALUES_PATH):
        with open(path, "r") as f:
            qvalues = json.load(f)
        print("Qvalues loaded")
        return qvalues

    def save_qvalues(self, path=configs.DEFAULT_QVALUES_PATH):
        with open(path, "w") as f:
            json.dump(self.qvalues, f)

    def act(self, direction, snake, food):
        state = self._get_state(direction, snake, food)

        # Epsilon greedy
        rand = random.uniform(0, 1)
        if rand < self.epsilon:
            action_key = random.choices(list(self.actions.keys()))[0]
        else:
            state_scores = self.qvalues[self._get_state_str(state)]
            action_key = state_scores.index(max(state_scores))
        action_val = self.actions[action_key]

        # Remember the actions it took at each state
        self.history.append({"state": state, "action": action_key})
        return action_val

    def update_q_values(self, reason):
        history = self.history[::-1]
        for i, h in enumerate(history[:-1]):
            if reason:  # Snake Died -> Negative reward
                sN = history[0]["state"]
                aN = history[0]["action"]
                state_str = self._get_state_str(sN)
                if reason == "Steps":  # Snake took too many steps
                    reward = -3
                else:
                    reward = -30
                self.qvalues[state_str][aN] = (1 - self.learning_rate) * self.qvalues[state_str][
                    aN
                ] + self.learning_rate * reward  # Bellman equation - there is no future state since game is over
                reason = None
            else:
                s1 = h["state"]  # current state
                s0 = history[i + 1]["state"]  # previous state
                a0 = history[i + 1]["action"]  # action taken at previous state

                x1 = s0.distance[0]  # x distance at current state
                y1 = s0.distance[1]  # y distance at current state

                x2 = s1.distance[0]  # x distance at previous state
                y2 = s1.distance[1]  # y distance at previous state

                if s0.food != s1.food:  # Snake ate a food, positive reward
                    reward = 10
                elif abs(x1) > abs(x2) or abs(y1) > abs(
                    y2
                ):  # Snake is closer to the food, positive reward
                    reward = 0.001
                else:
                    reward = -0.001  # Snake is further from the food, negative reward

                state_str = self._get_state_str(s0)
                new_state_str = self._get_state_str(s1)
                self.qvalues[state_str][a0] = (1 - self.learning_rate) * (
                    self.qvalues[state_str][a0]
                ) + self.learning_rate * (
                    reward + self.discount * max(self.qvalues[new_state_str])
                )  # Bellman equation

    def _get_state(self, direction, snake, food):
        snake_head = snake[-1]
        dist_x = food[0] - snake_head[0]
        dist_y = food[1] - snake_head[1]

        if dist_x > 0:
            pos_x = "1"  # Food is to the right of the snake
        elif dist_x < 0:
            pos_x = "0"  # Food is to the left of the snake
        else:
            pos_x = "NA"  # Food and snake are on the same X file

        if dist_y > 0:
            pos_y = "3"  # Food is below snake
        elif dist_y < 0:
            pos_y = "2"  # Food is above snake
        else:
            pos_y = "NA"  # Food and snake are on the same Y file

        adjacent_squares = [
            (snake_head[0] - self.block_size, snake_head[1]),
            (snake_head[0] + self.block_size, snake_head[1]),
            (snake_head[0], snake_head[1] - self.block_size),
            (snake_head[0], snake_head[1] + self.block_size),
        ]

        surrounding_list = []
        for square in adjacent_squares:
            if square[0] < 0 or square[1] < 0:  # off screen left or top
                surrounding_list.append("1")
            elif (
                square[0] >= self.display_width or square[1] >= self.display_height
            ):  # off screen right or bottom
                surrounding_list.append("1")
            elif square in snake[:-1]:  # part of tail
                surrounding_list.append("1")
            else:
                surrounding_list.append("0")
        surroundings = "".join(surrounding_list)

        return GameState((dist_x, dist_y), (pos_x, pos_y), surroundings, food, direction)

    def _get_state_str(self, state):
        return str(
            (state.position[0], state.position[1], state.surroundings, state.direction)
        )
