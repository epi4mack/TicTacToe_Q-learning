import numpy as np
import random

from env import TicTacToeEnv

class RandomAgent:
    def choose_action(self, env):
        return random.choice(env.available_actions())

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-таблица
        self.alpha = alpha  # скорость обучения
        self.gamma = gamma  # коэффициент дисконтирования
        self.epsilon = epsilon  # вероятность выбора случайного действия

    def choose_action(self, state, env):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(env.available_actions())
        
        # Выбрать действие с наивысшим Q-значением
        state_actions = self.q_table.get(state, {})
        max_value = max(state_actions.values(), default=0)
        best_actions = [action for action, value in state_actions.items() if value == max_value]
        
        return random.choice(best_actions) if best_actions else random.choice(env.available_actions())

    def update_q_value(self, state, action, reward, next_state):
        state_actions = self.q_table.setdefault(state, {})
        current_q = state_actions.get(action, 0)
        
        # Рассчитать новое Q-значение
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
        
        # Обновить Q-таблицу
        state_actions[action] = new_q

# Цикл обучения
def train(agent, opponent, env, episodes=1000):
    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Агент выбирает действие
            action = agent.choose_action(state, env)
            next_state, reward, done = env.step(action)
            
            # Оппонент случайно выбирает действие, если игра не окончена
            if not done:
                opp_action = opponent.choose_action(env)
                next_state, opp_reward, done = env.step(opp_action)
                reward = -opp_reward if opp_reward != 0 else reward  # Если оппонент выигрывает, агент получает негативное вознаграждение

            # Обновляем Q-значение
            agent.update_q_value(state, action, reward, next_state)
            state = next_state  # Переход к новому состоянию


if __name__ == '__main__': 
    env = TicTacToeEnv()
    agent = QLearningAgent()
    opponent = RandomAgent()

    train(agent, opponent, env, episodes=100)
