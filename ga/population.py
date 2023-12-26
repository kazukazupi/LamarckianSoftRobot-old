import os
from evogym import hashable
from ga.individual import Individual
from utils import Config

class Population:

    def __init__(self):
        
        self.individuals = []
        self.group_hashes = {}
        self.generation = 0
        self.num_evals = 0

        # sample robots
        for id in range(Config.population_size):
            
            individual = Individual.configure_new(id, Config.shape, self.get_saving_dir_name(id))

            while (hashable(individual.body) in self.group_hashes):
                individual = Individual.configure_new(id, Config.shape, self.get_saving_dir_name(id))

            self.group_hashes[hashable(individual.body)] = True
            individual.save()
            self.individuals.append(individual)
    
    # reproduce self.individuals[child_id] from parent by mutation
    def configure_from_mutation(self, child_id, parent):
        
        child = Individual.reproduce_by_mutation(parent, child_id, self.get_saving_dir_name(child_id))
        while (child is None) or (hashable(child.body) in self.group_hashes and self.group_hashes[hashable[child.body]]):
            child = Individual.reproduce_by_mutation(parent, child_id, self.get_saving_dir_name(child_id))
        child.save()
        
        self.group_hashes[hashable(child.body)] = True
        self.individuals[child_id] = child

    def get_saving_dir_name(self, id=int):
        return os.path.join(Config.exp_dir, f'generation{self.generation:02}/id{id:02}')
