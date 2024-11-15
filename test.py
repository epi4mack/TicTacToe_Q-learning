from time import perf_counter
from random import choice

from env import TicTacToeEnv
from QLearningAgent import QLearningAgent

def choose_random_action(env):
    action = choice(list(env.available_actions))
    env.available_actions.remove(action)
    return action


def train(agent, env, episodes=1000) -> None:
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # Агент выбирает действие
            action = agent.choose_action(state, env)
            next_state, reward, done = env.step(action)
            print('Агент')
            
            if done:
                # Если игра завершилась после хода обучающегося агента
                # reward будет 1 при победе агента, 0 при ничьей
                final_reward = reward  # Победа (+1) или ничья (0)
            else:
                # Если игра продолжается, случайный агент выбирает действие
                opp_action = choose_random_action(env)
                next_state, opp_reward, done = env.step(opp_action)
                print('рандом')
                
                if done:
                    # Если случайный агент завершил игру
                    final_reward = -1 if opp_reward != 0 else 0  # Поражение (-1) или ничья (0)
                else:
                    final_reward = 0  # Если игра продолжается

            # Обновление Q-значения для обучающегося агента
            agent.update_q_value(state, action, final_reward, next_state)
            state = next_state  # Переход к новому состоянию

        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        # print(f'Finished episode {episode + 1}')


def display_sorted_q_table(q_table):
    f = open('table.txt', 'a')

    print("Sorted Q-table:", file = f)
    print("=" * 40, file = f)
    
    for state, q_values in q_table.items():
        # Создаём список индексов и значений Q(s, a)
        actions_and_values = [(index, value) for index, value in enumerate(q_values)]
        # Сортируем по значению в порядке убывания
        sorted_actions = sorted(actions_and_values, key=lambda x: x[1], reverse=True)
        
        print(f"State: {state}", file = f)
        print("Actions sorted by Q-values:", file = f)
        for action_index, value in sorted_actions:
            # Преобразуем индекс действия обратно в (row, col)
            row, col = divmod(action_index, 3)
            print(f"  Action: ({row}, {col}), Q-value: {value:.3f}", file = f)
        print("-" * 40, file = f)

    f.close()



if __name__ == "__main__":

    env = TicTacToeEnv()

    alpha = 0.9 # Скорость обучения
    gamma = 0.95 # Коэффициент дисконтирования
    epsilon = 1 # Вероятность выбора случайного действия

    epsilon_decay = 0.99
    min_epsilon = 0.01

    learning_agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, min_epsilon)

    episodes = 10_000

    start_time = perf_counter()

    train(
        agent=learning_agent,
        env=env,
        episodes=episodes
    )

    total_time = perf_counter() - start_time
    print(f'Training time: {total_time:.2f} s')

    display_sorted_q_table(learning_agent.q_table)

    # for st in learning_agent.q_table:
    #     state = learning_agent.q_table[st]
    #     print(st)
    #     for i in state:
    #         print(i)
    #     print()
