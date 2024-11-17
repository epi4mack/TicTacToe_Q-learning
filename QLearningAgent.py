import numpy as np
import random

class QLearningAgent:
    def __init__(self, epsilon=0.05):
        self.q_table = {}  # Q-таблица
        self.epsilon = epsilon

    def choose_action(self, state, env):
        def convert_indices(cell_number):
            x = cell_number // 3
            y = cell_number % 3
            return x, y

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(env.available_actions))
            env.available_actions.remove(action)
            return action

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        max_value = np.max(self.q_table[state])

        best_actions = []
        for action in env.available_actions:
            idx = action[0] * 3 + action[1]
            if self.q_table[state][idx] == max_value:
                best_actions.append(action)

        final_choice = random.choice(best_actions)
        env.available_actions.remove(final_choice)

        return final_choice

    # def update_values(self, state, action, reward):
    #     idx = action[0] * 3 + action[1]
    #
    #     if state not in self.q_table:
    #         self.q_table[state] = np.zeros(9)
    #
    #     self.q_table[state][idx] = reward

    def update_values(self, state, action, reward, next_state, alpha=0.3, gamma=0.9):
        idx = action[0] * 3 + action[1]

        # Инициализация Q-таблицы для текущего и следующего состояний
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        # Текущее значение Q
        current_q = self.q_table[state][idx]

        # Максимальное Q-значение для следующего состояния
        max_next_q = np.max(self.q_table[next_state])

        # Применение формулы Q-обучения
        self.q_table[state][idx] = current_q + alpha * (reward + gamma * max_next_q - current_q)
