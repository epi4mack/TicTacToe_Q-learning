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
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(env.available_actions))
            env.available_actions.remove(action)
            return action
        
        # Выбираем лучшее действие на основе Q-значений (эксплуатация)
        q_values = self.get_q_values(state)
        max_value = -float('inf')
        best_actions = []

        for action in env.available_actions:
            idx = action[0] * 3 + action[1]
    
            if q_values[idx] > max_value:
                best_actions = [action]
                max_value = q_values[idx]
            elif q_values[idx] == max_value:
                best_actions.append(action)
        
        final_choice = random.choice(best_actions) if best_actions else random.choice(env.available_actions)
        env.available_actions.remove(final_choice)

        return final_choice

    def update_values(self, state, action, reward):
        # Обновляем ценности всех посещённых состояний
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        else:
            idx = action[0] * 3 + action[1]
            self.q_table[state][idx] = reward
