import numpy as np
import random

class QLearningAgent:
    def __init__(self, epsilon=0.05):
        self.q_table = {}
        self.epsilon = epsilon

    def choose_action(self, state, env):
        if self.epsilon and random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(env.available_actions))
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
        return final_choice

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
