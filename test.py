from utils import load_model
from env import TicTacToeEnv

from random import choice

def choose_random_action(env):
    action = choice(list(env.available_actions))
    return action

def test_model(env, agent, number=10000):
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
                ties += 1
                break

            if winner:
                wins += 1
                break

            # Ход человека
            state = next_state
            rand = choose_random_action(env)
            next_state, winner = env.step(rand)

            if winner == 0:
                ties += 1
                break

            if winner:
                loses += 1
                break

            state = next_state

    win_percentage = (wins / number) * 100
    lose_percentage = (loses / number) * 100
    tie_percentage = (ties / number) * 100

    print(f'\nПобед: {win_percentage:.2f}\nПоражений: {lose_percentage:.2f}\nНичьи: {tie_percentage:.2f}\n')


if __name__ == '__main__':
    env = TicTacToeEnv()

    model_name = 'model_5M.pickle'
    model = load_model(model_name)

    model.epsilon = 0
    test_model(env, model)
