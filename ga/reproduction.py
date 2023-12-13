import numpy as np

from evogym import get_uniform, draw, is_connected, has_actuator, get_full_connectivity


def mutate_structure(body, mutation_rate=0.1) -> tuple:

    pd = get_uniform(5)
    pd[0] = 0.6 # it is 3 times more likely for a cell to become empty

    body_return = body.copy()

    while True:
        for i in range(body_return.shape[0]):
            for j in range(body_return.shape[1]):
                mutation = [mutation_rate, 1 - mutation_rate]
                if draw(mutation) == 0:
                    body_return[i][j] = draw(pd)
        if is_connected(body_return) and has_actuator(body_return) and (not np.array_equal(body_return, body)):
            return body_return, get_full_connectivity(body_return)
        else: body_return = body.copy()
