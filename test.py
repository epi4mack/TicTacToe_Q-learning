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
        print(episode)
        state = env.reset()
        done = False

        while not done:
            # Агент выбирает действие
            action = agent.choose_action(state, env)

            next_state, reward, done = env.step(action)
            
            if done:
                final_reward = reward
            else:
                # Если игра продолжается, случайный агент выбирает действие
                opp_action = choose_random_action(env)
                _, opp_reward, done = env.step(opp_action)
                
                if opp_reward == 1:
                    final_reward = 0  # Агент проиграл
                elif opp_reward == 0.5:
                    final_reward = 0.5  # Ничья
                else:
                    final_reward = 0.5  # Агент продолжает играть (без завершения игры)

            # Обновление Q-значения для обучающегося агента
            # agent.update_q_table(state, action, final_reward)

            prev_state = state
            state = next_state  # Переход к новому состоянию
        
        print(final_reward)
        agent.update_values(prev_state, action, final_reward)



def display_sorted_q_table(q_table):
    f = open('table.txt', 'a')

    print("Sorted Q-table:", file=f)
    print("=" * 40, file=f)
    
    for state, q_values in q_table.items():
        # Создаём список индексов и значений Q(s, a)
        actions_and_values = [(index, value) for index, value in enumerate(q_values)]
        # Сортируем по значению в порядке убывания
        sorted_actions = sorted(actions_and_values, key=lambda x: x[1], reverse=True)
        
        print(f"State: {state}", file=f)
        print("Actions sorted by Q-values:", file=f)
        for action_index, value in sorted_actions:
            # Преобразуем индекс действия обратно в (row, col)
            row, col = divmod(action_index, 3)
            print(f"  Action: ({row}, {col}), Q-value: {value:.3f}", file=f)
        print("-" * 40, file=f)

    f.close()




if __name__ == "__main__":

    env = TicTacToeEnv()
    epsilon = 0.05 # Вероятность выбора случайного действия

    learning_agent = QLearningAgent(epsilon=epsilon)

    episodes = 1

    start_time = perf_counter()

    train(
        agent=learning_agent,
        env=env,
        episodes=episodes
    )

    total_time = perf_counter() - start_time
    print(f'Training time: {total_time:.2f} s')

    display_sorted_q_table(learning_agent.q_table)
