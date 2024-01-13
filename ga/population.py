import glob
import numpy as np
import os
from evogym import hashable
from ga.individual import Individual
from utils import Config


def get_saving_dir_name(generation:int, id:int):
    return os.path.join(Config.exp_dir, f'generation{generation:02}/id{id:02}')

class Population:

    def __init__(self, individuals, group_hashes, generation, num_evals):
        
        self.individuals = individuals
        self.group_hashes = group_hashes
        self.generation = generation
        self.num_evals = num_evals

    @staticmethod
    def initialize():

        individuals = []
        group_hashes = {}
        generation = 0
        num_evals = 0

        # sample robots
        for id in range(Config.population_size):
            
            individual = Individual.configure_new(id, Config.shape, get_saving_dir_name(generation, id))

            while (hashable(individual.body) in group_hashes):
                individual = Individual.configure_new(id, Config.shape, get_saving_dir_name(generation, id))

            group_hashes[hashable(individual.body)] = True
            individual.save()
            individuals.append(individual)

        return Population(individuals, group_hashes, generation, num_evals)

    @staticmethod
    def load():

        individuals = [None for i in range(Config.population_size)]
        group_hashes = {}
        generation = 0
        num_evals = 0

        while True:

            individual_dir_list = glob.glob(os.path.join(Config.exp_dir, f'generation{generation:02}/id*'))

            for individual_dir in individual_dir_list:
                individual = Individual.load(individual_dir)

                id = individual.id
                individuals[id] = individual
                group_hashes[hashable(individual.body)] = True

                if not individual.learning_en:
                    num_evals += 1

            if os.path.exists(os.path.join(Config.exp_dir, f'generation{(generation + 1):02}')):
                generation += 1
            else:
                break

        return Population(individuals, group_hashes, generation, num_evals)
    
    # reproduce self.individuals[child_id] from parent by mutation
    def configure_from_mutation(self, child_id, id_elite):

        count = 100

        while count > 0:
            parent_id = np.random.choice(id_elite)
            parent = self.individuals[parent_id]
            child = Individual.reproduce_by_mutation(parent, child_id, get_saving_dir_name(self.generation, child_id))
            if child is not None:
                if (not hashable(child.body) in self.group_hashes) or (not self.group_hashes[hashable(child.body)]):
                    self.group_hashes[hashable(child.body)] = True
                    self.individuals[child_id] = child
                    child.save()
                    return parent_id
            count -= 1

        return None

    # reproduce self.individuals[child_id] from two parents by crossover
    def configure_from_crossover(self, child_id, id_elite):

        count = 100
        
        while count > 0:
            parent1_id, parent2_id = np.random.choice(id_elite, size=2, replace=False)
            parent1 = self.individuals[parent1_id]
            parent2 = self.individuals[parent2_id]
            child = Individual.reproduce_by_crossover(parent1, parent2, child_id, get_saving_dir_name(self.generation, child_id))
            if child is not None:
                if (not hashable(child.body) in self.group_hashes) or (not self.group_hashes[hashable(child.body)]):
                    self.group_hashes[hashable(child.body)] = True
                    self.individuals[child_id] = child
                    child.save()
                    return parent1_id, parent2_id
            count -= 1

        return None

    
