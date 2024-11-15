import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # игровое поле 3x3, где 0 - пустая клетка
        self.current_player = 1#-1  # игрок 1 начинает первым (1 для X, -1 для O)
        self.available_actions = {(i, j) for i in range(3) for j in range(3)}

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1#-1
        self.available_actions = [(i, j) for i in range(3) for j in range(3)]
        return self.get_state()

    def get_state(self):
        # Возвращает состояние доски в виде одномерного массива
        return tuple(int(x) for x in self.board.flatten())

    def step(self, action):
        # Делает шаг игрока в выбранную клетку
        i, j = action
        self.board[i, j] = self.current_player

        d = {-1: '0', 1: 'X'}
        print(f'Ход делает {d[self.current_player]}')

        reward = self.check_winner()  # Вознаграждение: 1 - победа текущего игрока, 0 - ничья или продолжаем
        done = reward != 0 or not self.available_actions  # Игра окончена, если есть победитель или ничья
        self.current_player *= -1  # Переключаем игрока

        return self.get_state(), reward, done

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return 1 if self.current_player == 1 else -1
        if abs(self.board.trace()) == 3 or abs(np.fliplr(self.board).trace()) == 3:
            return 1 if self.current_player == 1 else -1
        return 0  # Победителя нет
