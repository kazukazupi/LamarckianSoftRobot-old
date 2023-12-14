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


def crossover(body1:np.ndarray, body2:np.ndarray) -> tuple:

    X = body1.shape[0]
    Y = body1.shape[1]

    count = 100

    while True:

        if count == 0:
            print('failed in connecting bodies')
            return None

        axis = np.random.choice([0, 1])
        axis = 1
        
        if axis == 0:
            mid = np.random.choice([y for y in range(1, Y)])
        else:
            mid = np.random.choice([x for x in range(1, X)])
        mid = 3

        if axis == 0:
            child_body = np.concatenate((body1[:mid], body2[mid:]), axis)
        else:
            child_body = np.concatenate((body1[:,:mid], body2[:,mid:]), axis)

        if is_connected(child_body) and has_actuator(child_body):
            break

        count -= 1

    return child_body, get_full_connectivity(child_body)