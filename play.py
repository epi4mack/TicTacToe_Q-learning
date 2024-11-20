from utils import load_model
from env import TicTacToeEnv

from os import system

def play_with_human(env, agent, number=1):
    for _ in range(number):
        state = env.reset()

        while True:
            # Ход крестиков (агент)
            agent_move = agent.choose_action(state, env)
            next_state, winner = env.step(agent_move)

            system('cls')
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

            system('cls')
            env.render()

            if winner == 0:
                print('\nНичья')
                break

            if winner:
                print('\nПобеда O')
                break

            state = next_state


if __name__ == '__main__':
    env = TicTacToeEnv()

    model_name = 'model_5M.pickle'
    model = load_model(model_name)

    play_with_human(env, model)
