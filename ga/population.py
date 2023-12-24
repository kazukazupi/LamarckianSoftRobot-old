import os

from ga.individual import Individual
from utils.config import Config

class Population:

    def __init__(self):
        
        self.individuals = {}
        self.generation = 0

        for id in range(Config.population_size):
            self.individuals[id] = Individual.configure_new(
                id=id,
                shape=Config.shape,
                saving_dir=self.get_saving_dir_name(id)
            )

    def append_individual(self, individual: Individual):
        
        id = individual.id
        assert not id in self.individuals.keys()
        self.individuals[id] = individual

    def remove_individual(self, id:int):
        del self.individuals[id]

    def get_saving_dir_name(self, id=int):
        return os.path.join(Config.exp_dir, f'generation{self.generation:02}/id{id:02}')
