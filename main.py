from time import perf_counter
from random import choice
import matplotlib.pyplot as plt

from env import TicTacToeEnv
from QLearningAgent import QLearningAgent

from utils import save_model, load_model

def choose_random_action(env):
    action = choice(list(env.available_actions))
    return action

def train(agent, env, episodes=1000, alpha=0.4, gamma=0.9, interval=8000) -> None:
    x = []
    y = []

    cumulative_reward = 0
    for episode in range(episodes):

        if episode % interval == 0:
            x.append(episode)
            avg_reward = cumulative_reward / interval
            y.append(avg_reward)
            cumulative_reward = 0

        state = env.reset()

        while True:
            # Агент выбирает и делает ход
            action = agent.choose_action(state, env)
            next_state, winner, = env.step(action)

            # Если есть победитель, либо ничья
            if winner is not None:
                final_reward = 0.5 if not winner else 1
                agent.update_values(state, action, final_reward, next_state, alpha, gamma)
                state = next_state  # Переход к новому состоянию
                cumulative_reward += final_reward
                break

           # Игра продолжается, оппонент делает ход
            opp_action = choose_random_action(env)
            next_state, winner = env.step(opp_action)

            if winner is not None:
                final_reward = 0.5 if not winner else 0
                agent.update_values(state, action, final_reward, next_state, alpha, gamma)
                state = next_state  # Переход к новому состоянию
                cumulative_reward += final_reward
                break

            # Оппонент сделал ход но игра все ещё продолжается
            final_reward = 0.5
            agent.update_values(state, action, final_reward, next_state, alpha, gamma)
            state = next_state  # Переход к новому состоянию
            cumulative_reward += final_reward

    return x, y


def draw_graph(x: list[int], y: list[float]) -> None:
    plt.plot(x, y, color='b', label='Зависимость награды от числа шагов')
    plt.xlabel("Шаги")
    plt.ylabel("Награда")

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    env = TicTacToeEnv()
    epsilon =  0.05 # Вероятность выбора случайного действия

    alpha = 0.7
    gamma = 0.1

    learning_agent = QLearningAgent(epsilon=epsilon)

    episodes = 100_000
    start_time = perf_counter()

    steps, rewards = (
        train(
        agent=learning_agent,
        env=env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        interval=episodes // 100
    ))

    total_time = perf_counter() - start_time
    print(f'Training time: {total_time:.2f} s')

    draw_graph(steps, rewards)

    # Сохраняем обученную модель, потом загружаем её из файла
    model_name = 'model_5M.pickle'
    # save_model(learning_agent, model_name)
