from types import FunctionType
import numpy as np
import copy
import random
from evolutionary_algorithm import EvolutionaryAlgorithm

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

def inverse_distance(individual):
  distance = 0
  num_individuals = len(individual)
  for i in range(len(individual)):
    distance += get_distance(city_dict[individual[i]], city_dict[individual[(i+1) % num_individuals]])
  return 1/distance

class Individual():
  def __init__(self, genome):
    self.genome = genome
    self.fitness = inverse_distance(self.genome)


def random_intercity_paths(population_size: int):
  population = []
  for i in range(population_size):
    genome = copy.deepcopy(city_list)
    np.random.shuffle(genome)
    population.append(Individual(genome))
  return population


def fitness_proportional_selection(population: list, num_selections: int):
  population_proportions = {}
  cumulative_fitness = 0
  best_individual = population[0]
  for individual in population:
    if(individual.fitness > best_individual.fitness):
      best_individual = individual
    cumulative_fitness += individual.fitness
    population_proportions[individual] = cumulative_fitness

  total_fitness = cumulative_fitness
  average_fitness = total_fitness/len(population)
  selections = []
  for i in range(num_selections):
    random_float = random.uniform(0, total_fitness)
    for i in population_proportions:
      #as soon as we find the first parent whose proportion starts after the random float, we append the parent before it to parents
      if(population_proportions[i] >= random_float):
        selections.append(i)
  return best_individual, average_fitness, selections


def tournament_selection(population: list, num_selections :int):
    pass

def rank_selection(population: list, num_selections :int):
   pass

def random_length_crossover(parent1: Individual, parent2: Individual) -> tuple:
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

    offspring1 = Individual(offspring1)
    offspring2 = Individual(offspring2)

    return offspring1, offspring2


def random_two_gene_swap(individual :Individual, mutation_rate: float):
  rand_numb = random.randint(0,100)/100
  if rand_numb <= mutation_rate:
      # mutation defined by reversing orders for now and to be changed
      # porposed algorithm would be to randomly swich 10 neighboring places such that it's neighbors have less distance as compared with the one
      rand_index1 = random.randint(0, len(individual.genome)-1)
      rand_index2 = random.randint(0, len(individual.genome)-1)

      individual.genome[rand_index1], individual.genome[rand_index2] = individual.genome[rand_index2], individual.genome[rand_index1]

  return Individual(individual.genome)


def random_selection(population: list, num_selections: int):
  survivors = []
  for i in range(num_selections):
    random_int = random.randint(0, len(population)-1)
    survivors.append(population[random_int])
  return 0,0,survivors

def truncation_selection(population, size):
  result = []
  result = copy.deepcopy(population)
  result.sort(key=lambda k : k.fitness, reverse=True)
  # print(result)
  return result[0], 0, result[:size]


tsp = EvolutionaryAlgorithm(
    fitness_function = inverse_distance,
    initial_population_function = random_intercity_paths,
    parent_selection_function = truncation_selection,
    survivor_selection_function = random_selection,
    cross_over_function = random_length_crossover,
    mutation_operator = random_two_gene_swap,
    population_size = 100,
    mutation_rate = 0.5
)

tsp.run()