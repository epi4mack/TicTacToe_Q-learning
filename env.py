import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # игровое поле 3x3, где 0 - пустая клетка
        self.current_player = 1 # игрок 1 начинает первым (1 для X, -1 для O)
        self.available_actions = {(i, j) for i in range(3) for j in range(3)}

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.available_actions = [(i, j) for i in range(3) for j in range(3)]
        return self.get_state()

    def get_state(self):
        # Возвращает состояние доски в виде одномерного массива
        return tuple(int(x) for x in self.board.flatten())

    def step(self, action):
        i, j = action

        if self.board[i][j]:
            raise ValueError('Клетка занята!')

        self.board[i, j] = self.current_player
        self.available_actions.remove((i, j))

        winner = self.check_winner()
        self.current_player *= -1

        return self.get_state(), winner

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return self.current_player
        if abs(self.board.trace()) == 3 or abs(np.fliplr(self.board).trace()) == 3:
            return self.current_player
        if not self.available_actions:
            return 0
        return None

    def make_human_move(self):
        while True:
            action = input('\nХод: ').split()

            try:
                x, y = action[0], action[1]
            except:
                continue

            if not (x.isdigit() and y.isdigit()):
                continue

            x, y = int(x) - 1, int(y) - 1
            valid_range = set(range(3))
            if not (x in valid_range and y in valid_range):
                continue

            if self.board[x][y]:
                continue

            return self.step((x, y))


    def render(self):
        d = {-1: 'O', 1: 'X', 0: ' '}

        for row in self.board:
            for i, col in enumerate(row):
                print(f'{d[col]}', end='|')
            print()
