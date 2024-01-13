import csv
import numpy as np
import os

from utils import Config, print_and_save
from ga.population import Population

def run_ga():

    #-------------------------
    # 1. setup
    #-------------------------
    
    # read args
    Config.initialize()
    
    # set experiment directory
    os.makedirs(Config.exp_dir)
    Config.dump(os.path.join(Config.exp_dir, 'config.json'))

    # txt file to log
    log_file_path = os.path.join(Config.exp_dir, 'log.txt')
    print_and_save(f'started evolution.', log_file_path, mode='w')
    print_and_save(f'save results in : {Config.exp_dir}', log_file_path)

    # csv file to log fitness
    fitness_csv_path = os.path.join(Config.exp_dir, 'fitness.csv')
    with open(fitness_csv_path, 'w') as f:
        writer = csv.writer(f)
        row = ['i_generation'] + [f'id{id}' for id in range(Config.population_size)]
        writer.writerow(row)
    
    # initialize population
    population = Population.initialize()
    print_and_save(f'Initialized {Config.population_size} individuals.', log_file_path)

    #-------------------------
    # 2. evolution
    #-------------------------

    while True:

        print_and_save("============================================================================================", log_file_path)
        print_and_save(f'i_generation: {population.generation}', log_file_path)
        print_and_save("============================================================================================", log_file_path)

        #-------------------------
        # 2-1. learning
        #-------------------------

        # learning of newborn robots
        for id, individual in enumerate(population.individuals):
            
            # train if robot has not leaned
            if individual.learning_en:
                
                print_and_save(f'training individual {id} (parents: {individual.parents_id})...', log_file_path)

                # train and get fitness
                parents = [population.individuals[parent_id] for parent_id in individual.parents_id]
                individual.train(parents)
                print_and_save(f'\tterminated. fitness:{individual.fitness}', log_file_path)

                individual.learning_en = False
                individual.save()

                population.num_evals += 1

                # end evolution
                if (population.num_evals == Config.max_evaluations):
                    break
            
            # skip training if robot has already learned
            else:
                print_and_save(f'skip training individual {id} (parents: {individual.parents_id}).', log_file_path)

        # log fitness values to csv
        with open(fitness_csv_path, 'a') as f:
            writer = csv.writer(f)
            row = [population.generation] + [individual.fitness for individual in population.individuals]
            writer.writerow(row)

        # end evolution
        if population.num_evals == Config.max_evaluations:
            print_and_save(f'end evolution.', log_file_path)
            return
        
        #-------------------------
        # 2-2. genetic operation
        #-------------------------

        population.generation += 1
        elite_rate = (Config.max_evaluations - population.num_evals - 1) / (Config.max_evaluations - 1) * (Config.elite_rate_high - Config.elite_rate_low) + Config.elite_rate_low
        num_survivors = int(max([2, np.ceil(elite_rate * Config.population_size)]))

        print_and_save("--------------------------------------------------------------------------------------------", log_file_path)
        print_and_save(f'genetic operation (elite_rate: {elite_rate}, num_survivors: {num_survivors})', log_file_path)
        print_and_save("--------------------------------------------------------------------------------------------", log_file_path)

        fitness_array = np.array([individual.fitness for individual in population.individuals])
        id_ranking = np.argsort(-fitness_array)
        id_elite = id_ranking[0 : num_survivors]

        print_and_save(f'elites: {id_elite}.', log_file_path)

        for id, individual in enumerate(population.individuals):

            # keep if robot is elite
            if id in id_elite:
                assert not individual.learning_en

            else:
                while True:
                    # configure newborn robot from crossover
                    if np.random.random() < Config.crossover_rate:
                        result = population.configure_from_crossover(id, id_elite)
                        if result is not None:
                            parent1_id, parent2_id = result
                            print_and_save(f'individual {id} was reproduced from {parent1_id}, {parent2_id} by crossover', log_file_path)
                            break
                    # configure newborn robot from mutation
                    else:
                        result = population.configure_from_mutation(id, id_elite)
                        if result is not None:
                            parent1_id = result
                            parent_id = population.configure_from_mutation(id, id_elite)
                            print_and_save(f'individual {id} was reproduced from {parent_id} by mutation', log_file_path)
                            break
