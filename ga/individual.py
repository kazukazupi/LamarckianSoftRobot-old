import os
import json
import numpy as np
from evogym import sample_robot
from ga.reproduction import mutate_structure, crossover
from ppo import run_ppo


JSON_FILE_NAME = 'robot_info.json'
BODY_FILE_NAME = 'body.npy'
CONNECTIONS_FILE_NAME = 'connections.npy'


class Individual:

    def __init__(
            self,
            id:int,
            body:np.ndarray,
            connections:np.ndarray,
            saving_dir:str,
            parents_id,
            learning_en=True,
            fitness=None,
            crossover_info=None
        ):

        assert len(parents_id) <= 2

        # initialize
        self.id = id
        self.body = body
        self.connections = connections
        self.saving_dir = saving_dir
        self.learning_en = learning_en
        self.fitness = fitness
        self.parents_id = parents_id
        self.crossover_info = crossover_info

        # make directory
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

    def train(self, parents):
        
        # tarin by ppo
        self.fitness = run_ppo(
            body=self.body,
            connections=self.connections,
            saving_dir=self.saving_dir,
            parents=parents,
            crossover_info=self.crossover_info
        )

    def save(self):

        # save structure
        np.save(os.path.join(self.saving_dir, BODY_FILE_NAME), self.body)
        np.save(os.path.join(self.saving_dir, CONNECTIONS_FILE_NAME), self.connections)

        # save information in robot_info.json
        dict_to_write = {
            'id': self.id,
            'learning_en': self.learning_en,
            'parents_id': self.parents_id,
            'fitness': self.fitness
        }
        json_path = os.path.join(self.saving_dir, JSON_FILE_NAME)

        with open(json_path, 'w') as fp:
            json.dump(dict_to_write, fp)

    @staticmethod
    def load(saving_dir:str):

        # load structure
        body = np.load(os.path.join(saving_dir, BODY_FILE_NAME))
        connections = np.load(os.path.join(saving_dir, CONNECTIONS_FILE_NAME))

        # load information
        with open(os.path.join(saving_dir, JSON_FILE_NAME)) as fp:
            read_dict = json.load(fp)

        id = read_dict['id']
        learning_en = read_dict['learning_en']
        parents_id = read_dict['parents_id']
        fitness = read_dict['fitness']

        return Individual(
            id,
            body,
            connections,
            saving_dir,
            parents_id,
            learning_en=learning_en,
            fitness=fitness
        )
    
    @staticmethod
    def configure_new(id:int, shape:tuple, saving_dir:str):

        body, connections = sample_robot(shape)
        return Individual(id, body, connections, saving_dir, [])

    @staticmethod
    def reproduce_by_mutation(parent, id:int, saving_dir:str):

        result = mutate_structure(parent.body)

        if result is None:
            return None
        else:
            (body, connections) = result
            return Individual(id, body, connections, saving_dir, [parent.id])
    
    @staticmethod
    def reproduce_by_crossover(parent1, parent2, id:int, saving_dir:str):

        result = crossover(parent1.body, parent2.body)

        if result is None:
            return None
        else:
            (body, connections), (axis, mid) = result
            crossover_info = {'axis': axis, 'mid': mid}
            return Individual(id, body, connections, saving_dir, [parent1.id, parent2.id], crossover_info=crossover_info)
