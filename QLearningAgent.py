import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.05):
        self.q_table = {}  # Q-таблица
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.epsilon = epsilon

    def get_q_values(self, state):
        # Получить Q-значения для данного состояния или инициализировать, если оно новое
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)  # Инициализация для 9 возможных действий 
        return self.q_table[state]

    def choose_action(self, state, env):
        def convert_indices(cell_number):
            x = cell_number // 3
            y = cell_number % 3
            return (x, y)

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(env.available_actions))
            env.available_actions.remove(action)
            return action

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        max_value = (max(self.q_table[state]))

        best_actions = []
        for action in env.available_actions:
            idx = action[0] * 3 + action[1]
            if self.q_table[state][idx] == max_value:
                best_actions.append(action)

        final_choice = random.choice(best_actions)
        env.available_actions.remove(final_choice)

        # Выбираем лучшее действие на основе Q-значений (эксплуатация)
        # q_values = self.get_q_values(state)
        # max_value = -float('inf')
        # best_actions = []
        #
        # for action in env.available_actions:
        #     idx = action[0] * 3 + action[1]
        #
        #     if q_values[idx] > max_value:
        #         best_actions = [action]
        #         max_value = q_values[idx]
        #     elif q_values[idx] == max_value:
        #         best_actions.append(action)
        #
        # final_choice = random.choice(best_actions) if best_actions else random.choice(env.available_actions)
        # env.available_actions.remove(final_choice)

        return final_choice

    def update_values(self, state, action, reward):
        idx = action[0] * 3 + action[1]

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        self.q_table[state][idx] = reward
