import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01):
        self.q_table = {}  # Q-таблица
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.epsilon = epsilon  # Вероятность случайного действия
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

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

    # def update_q_value(self, state, action, reward, next_state):
    #     q_values = self.get_q_values(state)
    #     idx = action[0] * 3 + action[1]
        
    #     # Рассчёт нового Q-значения
    #     current_q = q_values[idx]
    #     next_max = max(self.get_q_values(next_state))
    #     new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
        
    #     # Обновляем Q-значение в таблице
    #     q_values[idx] = new_q

    def update_q_value(self, state, action, reward, next_state):

        # Если текущее или следующее состояние не в Q-таблице, инициализируем их
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        # Индекс действия
        action_index = action[0] * 3 + action[1]

        # Старое значение Q
        old_value = self.q_table[state][action_index]

        # Максимальное Q-значение для следующего состояния
        next_max = max(self.q_table[next_state])

        # Применяем формулу обновления
        self.q_table[state][action_index] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
