from types import FunctionType
from typing import List
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
import copy

class Individual():
  def __init__(self, genome, fitness):
    self.genome = genome
    self.fitness = fitness

  @abstractmethod
  def mutate(self) -> None:
    pass


class EvolutionaryAlgorithm():
  def __init__(self, 
               initial_population_function: FunctionType,
               parent_selection_function: str,
               survivor_selection_function: str,
               cross_over_function: FunctionType,
               population_size: int = 100,
               mutation_rate: float = 0.5,
               num_offsprings: int = 10,):
    
    selection_functions_string_map = {'truncation': self.truncation_selection,
                                      'random': self.random_selection,
                                      'binary': self.binary_tournament_selection}

    self.initial_population_function: FunctionType = initial_population_function
    self.population: List[Individual] = initial_population_function(population_size)
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.parent_selection_function: FunctionType = selection_functions_string_map[parent_selection_function]
    self.survivor_selection_function: FunctionType = selection_functions_string_map[survivor_selection_function]
    self.cross_over_function: FunctionType = cross_over_function
    self.num_offsprings: int = num_offsprings


  ## selection functions
  def random_selection(self, num_selections: int) -> List[Individual]:
    survivors = []
    for i in range(num_selections):
      random_int = random.randint(0, len(self.population)-1)
      survivors.append(self.population[random_int])
    return survivors


  def truncation_selection(self, num_selections: int) -> List[Individual]:
    result = []
    result = copy.deepcopy(self.population)
    result.sort(key=lambda k : k.fitness)
    return result[:num_selections]
  

  def binary_tournament_selection(self, num_selections: int) -> List[Individual]:
    result= []
    for i in range(num_selections):
        ind1, ind2 = random.sample(self.population, 2)
        selected = ind1 if ind1.fitness < ind2.fitness else ind2
        result.append(selected)
    return result

  def rank_selection(self, num_selections: int):
    rank = sorted(self.population, key=lambda x: x.fitness)
    total_rank = (len(rank) * (len(rank)  + 1)) / 2
    normalized_range = []
    result = []
    
    normalized_range.append(1/total_rank)

    for i in range(1,len(rank)):
        normalized_value = (i+1)/total_rank
        normalized_range.append(normalized_range[i-1] + normalized_value)
    

    for i in range(num_selections):
        random_numb = random.random()

        for i in range(len(normalized_range)):
            if random_numb < i:
                result.append(rank[i])
                break
    
    # print(normalized_range)
    return result

  def get_average_and_best_individual(self) -> (Individual, float):
    best_individual = self.population[0]
    cumulative_fitness = 0
    for individual in self.population:
      if(individual.fitness < best_individual.fitness):
        best_individual = individual
      cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/len(self.population)
    return best_individual, average_fitness



  def run_generation(self) -> None:
    parents = self.parent_selection_function(self.num_offsprings)

    # creating offspring
    for k in range(0, self.num_offsprings-1, 2):
      offspring1, offspring2 = self.cross_over_function(parents[k], parents[k+1])
      rand_num1, rand_num2 = random.randint(0,100)/100, random.randint(0,100)/100
      if rand_num1 <= self.mutation_rate:
        offspring1.mutate()
      if rand_num2 <= self.mutation_rate:
        offspring2.mutate()
      self.population.extend([offspring1, offspring2])

    self.population = self.survivor_selection_function(self.population_size)
        

  @abstractmethod
  def run():
    pass
    