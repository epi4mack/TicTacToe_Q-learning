from time import perf_counter
from random import choice
from os import system

from env import TicTacToeEnv
from QLearningAgent import QLearningAgent

def choose_random_action(env):
    action = choice(list(env.available_actions))
    env.available_actions.remove(action)
    return action

def train(agent, env, episodes=1000, alpha=0.4, gamma=0.9) -> None:
    for episode in range(episodes):
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
                break

           # Игра продолжается, оппонент делает ход
            opp_action = choose_random_action(env)
            next_state, winner = env.step(opp_action)

            if winner is not None:
                final_reward = 0.5 if not winner else 0
                agent.update_values(state, action, final_reward, next_state, alpha, gamma)
                state = next_state  # Переход к новому состоянию
                break

            # Оппонент сделал ход но игра все ещё продолжается
            final_reward = 0.5
            agent.update_values(state, action, final_reward, next_state, alpha, gamma)
            state = next_state  # Переход к новому состоянию


def display_sorted_q_table(q_table):
    def state_to_table(state):
        d = {0: '*', 1: 'X', -1: 'O'}

        row1 = [d[v] for v in state[:3]]
        row2 = [d[v] for v in state[3:6]]
        row3 = [d[v] for v in state[6:9]]

        result = f'\n\n{str(row1)}\n{str(row2)}\n{str(row3)}\n'

        return result

    f = open('table.txt', 'a')

    print("Sorted Q-table:", file=f)
    print("=" * 40, file=f)
    
    for state, q_values in q_table.items():
        # Создаём список индексов и значений Q(s, a)
        actions_and_values = [(index, value) for index, value in enumerate(q_values)]
        # Сортируем по значению в порядке убывания
        sorted_actions = sorted(actions_and_values, key=lambda x: x[1], reverse=True)
        
        print(f"State: {state_to_table(state)}", file=f)
        print("Actions sorted by Q-values:", file=f)
        for action_index, value in sorted_actions:
            # Преобразуем индекс действия обратно в (row, col)
            row, col = divmod(action_index, 3)
            print(f"  Action: ({row}, {col}), Q-value: {value:.3f}", file=f)
        print("-" * 40, file=f)

    f.close()


def play_with_human(env, agent, number=10000):
    system('cls')

    for _ in range(number):
        state = env.reset()

        while True:
            # Ход крестиков (агент)
            agent_move = agent.choose_action(state, env)
            next_state, winner = env.step(agent_move)

            env.render()

            if winner == 0:
                print('\nНичья')
                break

            if winner:
                print('\nПобеда X')
                break

            # Ход человека
            state = next_state
            next_state, winner = env.make_human_move()

            env.render()

            if winner == 0:
                print('\nНичья')
                break

            if winner:
                print('\nПобеда O')
                break

            state = next_state


    # Симуляция игр агента против случайного выьбора
def test_agent(env, agent, number=10000):
    wins = 0
    ties = 0
    loses = 0

    for _ in range(number):
        state = env.reset()

        while True:

            # Случайное действие крестиков
            agent_move = agent.choose_action(state, env)
            next_state, winner = env.step(agent_move)

            # env.render()

            if winner == 0:
                # print('\nНичья')
                ties += 1
                break

            if winner:
                # print('\nПобеда X')
                wins += 1
                break

            # Ход человека
            state = next_state
            rand = choose_random_action(env)
            next_state, winner = env.step(rand)
            # next_state, winner = env.make_human_move()

            # env.render()

            if winner == 0:
                # print('\nНичья')
                ties += 1
                break

            if winner:
                # print('\nПобеда O')
                loses += 1
                break

            state = next_state

    win_percentage = (wins / number) * 100
    lose_percentage = (loses / number) * 100
    tie_percentage = (ties / number) * 100

    print(f'\nПобед: {win_percentage:.2f}\nПоражений: {lose_percentage:.2f}\nНичьи: {tie_percentage:.2f}\n')



if __name__ == "__main__":

    env = TicTacToeEnv()
    epsilon = 0.05 # Вероятность выбора случайного действия

    alpha = 0.8
    gamma = 0.5

    learning_agent = QLearningAgent(epsilon=epsilon)

    episodes = 100_000
    start_time = perf_counter()

    train(
        agent=learning_agent,
        env=env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma
    )

    total_time = perf_counter() - start_time
    print(f'Training time: {total_time:.2f} s')

    # display_sorted_q_table(learning_agent.q_table)
    # print(len(learning_agent.q_table))

    learning_agent.epsilon = 0 # При игре против человека агент не будет случайно выбирать ход

    test_agent(env, learning_agent, number=50_000)

    play_with_human(env, learning_agent)
