from types import FunctionType
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm


'''Fetching Data'''
def read_and_convert_to_dict(file_path):
    data_dict = {}
    city_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into parts
            parts = line.strip().split()

            try:
              # Extract key and coordinates
              key = int(parts[0])
              # adding the city to the list as well to keep record
              city_list.append(key)
              coordinates = tuple(map(float, parts[1:]))
              # Create dictionary entry
              data_dict[key] = coordinates
            except:
              continue
    return city_list, data_dict
file_path = 'data/qa194.tsp'  # Replace with the path to your text file
city_list, city_dict = read_and_convert_to_dict(file_path)


def get_distance(x: tuple, y: tuple):
  # x and y are two 2D points each not 2 coordinates of one 2D point.
  return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**(1/2)

def distance(individual):
  distance = 0
  num_individuals = len(individual)
  for i in range(len(individual)):
    distance += get_distance(city_dict[individual[i]], city_dict[individual[(i+1) % num_individuals]])
  return distance


class TSP_Path(Individual):
  def __init__(self, genome):
    fitness = distance(genome)
    super().__init__(genome, fitness)
  
  def mutate(self) -> None:
    # mutation defined by reversing orders for now and to be changed
    # porposed algorithm would be to randomly swich 10 neighboring places such that it's neighbors have less distance as compared with the one
    rand_index1 = random.randint(0, len(self.genome)-1)
    rand_index2 = random.randint(0, len(self.genome)-1)

    self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]


def random_intercity_paths(population_size: int) -> List[TSP_Path]:
  population = []
  for i in range(population_size):
    genome = copy.deepcopy(city_list)
    np.random.shuffle(genome)
    population.append(TSP_Path(genome))
  return population


def TSP_random_length_crossover(parent1: TSP_Path, parent2: TSP_Path):
    start = random.randint(1, int(194/2))
    end = random.randint(int(194/2), 192)

    offspring1 = [None] * 194
    offspring2 = [None] * 194

    offspring1[start:end+1] = parent1.genome[start:end+1]
    offspring2[start:end+1] = parent2.genome[start:end+1]

    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1

    while None in offspring1:
        if parent2.genome[parent2_pointer] not in offspring1:
            offspring1[pointer % 194] = parent2.genome[parent2_pointer]
            pointer += 1
        parent2_pointer = (parent2_pointer + 1) % len(parent2.genome)

    pointer = 0

    while None in offspring2:
        if parent1.genome[parent1_pointer] not in offspring2:
            offspring2[pointer] = parent1.genome[parent1_pointer]
            pointer += 1
        parent1_pointer = (parent1_pointer + 1) % len(parent1.genome)

    offspring1 = TSP_Path(offspring1)
    offspring2 = TSP_Path(offspring2)

    return offspring1, offspring2


class TSP_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
   def run(self, num_iterations: int=10, num_generations: int=10000):
      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation()
          if(i % 100 == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("Average fitness: ", average_fitness, ", Best value: ", best_individual.fitness)

        self.population = self.inital_population_function()


tsp = TSP_EvolutionaryAlgorithm(
    initial_population_function = random_intercity_paths,
    parent_selection_function = 'fitness',
    survivor_selection_function = 'binary',
    cross_over_function = TSP_random_length_crossover,
    population_size = 100,
    mutation_rate = 0.5,
    num_offsprings=100
)
tsp.run(num_generations=10000)

''' Generating Graphs '''
selection_pairs = {('fitness', 'random'),
                    ('binary', 'truncation'), 
                    ('truncation', 'truncation'), 
                    ('random', 'random'),
                    ('fitness', 'rank'),
                    ('rank', 'binary')}