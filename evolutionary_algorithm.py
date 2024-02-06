from types import FunctionType
from tqdm import tqdm
import random


class EvolutionaryAlgorithm():
  def __init__(self, 
               fitness_function: FunctionType,
               initial_population_function: FunctionType,
               parent_selection_function: FunctionType,
               survivor_selection_function: FunctionType,
               cross_over_function: FunctionType,
               mutation_operator: FunctionType,
               population_size: int = 100,
               mutation_rate: float = 0.5):
    
    self.population: list = initial_population_function(population_size)
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.fitness_function: FunctionType = fitness_function
    self.parent_selection_function: FunctionType = parent_selection_function
    self.survivor_selection_function: FunctionType = survivor_selection_function
    self.mutation_operator: FunctionType = mutation_operator
    self.cross_over_function: FunctionType = cross_over_function

  def run_generation():
    pass

  def run(self, num_generations=4000, num_iterations=10):
    for j in range(num_iterations):
      for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
        # hardcoding number of parents to be selected to 2
        best_individual, average_fitness, parents = self.parent_selection_function(self.population, 10)

        #------ trash code -------#
        for k in range(0, len(parents)-1, 2):
          crossover1, crossover2 = self.cross_over_function(parents[k], parents[k+1])
          offspring1 = self.mutation_operator(crossover1, self.mutation_rate)
          offspring2 = self.mutation_operator(crossover2, self.mutation_rate)

          self.population.extend([offspring1, offspring2])
        #-------------------------#

        _, _, self.population = self.survivor_selection_function(self.population, self.population_size)
        if(i % 100 == 0):
          print("Average fitness: ", average_fitness, ", Best value: ", 1/best_individual.fitness)
          best_individual.save("data/fake_monalisa/fake_monalisa_"+str(j)+"_"+str(i)+".png")