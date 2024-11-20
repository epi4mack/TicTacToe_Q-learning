import pickle

def save_model(model, name: str = None) -> None:
    name = 'model.pickle' if name is None else name

    with open(name, 'wb') as file:
        pickle.dump(model, file)


def load_model(path: str = 'model.pickle'):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


def state_to_table(state: tuple):
    d = {0: '*', 1: 'X', -1: 'O'}

    row1 = [d[v] for v in state[:3]]
    row2 = [d[v] for v in state[3:6]]
    row3 = [d[v] for v in state[6:9]]

    result = f'\n\n{str(row1)}\n{str(row2)}\n{str(row3)}\n'
    return result


def display_sorted_q_table(q_table):
    with open('table.txt', 'w') as f:
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
