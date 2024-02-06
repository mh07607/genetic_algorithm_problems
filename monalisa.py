from PIL import Image, ImageDraw
import imageio
from evolutionary_algorithm import EvolutionaryAlgorithm
from types import FunctionType
import numpy as np
import copy
import random

# Initialize Pygame

# Set up the screen
width, height = 800, 800

# Load the reference image
reference_image = np.array(Image.open('data/monalisa.jpg'))

color_pallete = [(92, 75, 48), 
                 (41, 25, 28), 
                 (159, 161, 110), 
                 (174, 159, 90), 
                 (146, 104, 46),
                 (0,0,0)]


def draw_polygon(draw, color, vertices) -> None:
    draw.polygon(vertices, fill=color)

def generate_random_polygon():
    num_vertices = np.random.randint(3, 7)  # Random number of vertices (3 to 6)
    vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_vertices)]
    color = color_pallete[np.random.randint(0, 5)]  # Random RGB color
    return {'vertices': vertices, 'color': color}

def inverse_image_difference(genome):
    image = Image.new("RGB", (width, height), color=(159, 161, 110))
    draw = ImageDraw.Draw(image)
    # Draw each polygon
    for polygon in genome:
        draw_polygon(draw, polygon['color'], polygon['vertices'])
    # Capture the current screen as an image
    # Calculate the absolute pixel-wise difference
    diff = np.abs(reference_image - np.array(image))
    # Calculate the mean difference as the fitness score
    image_difference = np.mean(diff)
    return 1/image_difference


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = inverse_image_difference(self.genome)

    def save(self, image_name):
        image = Image.new("RGB", (width, height), color=(159, 161, 110))
        draw = ImageDraw.Draw(image)
        # Draw each polygon
        for polygon in self.genome:
            draw_polygon(draw, polygon['color'], polygon['vertices'])
        
        image.save(image_name)


def random_polygon_combinations(population_size: int) -> list:
    population = []
    for i in range(population_size):
        genome = [generate_random_polygon() for _ in range(100)]
        population.append(Individual(genome))
    return population


def random_length_crossover(parent1: Individual, parent2: Individual) -> tuple:
    start = random.randint(0, int(100/2))
    end = random.randint(int(100/2), 98)

    offspring1 = [None] * 100
    offspring2 = [None] * 100

    offspring1[start:end+1] = parent1.genome[start:end+1]
    offspring2[start:end+1] = parent2.genome[start:end+1]

    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1

    while None in offspring1:
        #if parent2.genome[parent2_pointer] not in offspring1:
        offspring1[pointer % 100] = parent2.genome[parent2_pointer]
        pointer += 1
        parent2_pointer = (parent2_pointer + 1) % len(parent2.genome)

    pointer = 0

    while None in offspring2:
        #if parent1.genome[parent1_pointer] not in offspring2:
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
      rand_coordinate = random.randint(0, 2)
      individual.genome[rand_index1]['vertices'][rand_coordinate], individual.genome[rand_index2][rand_coordinate] = individual.genome[rand_index2]['vertices'][rand_coordinate], individual.genome[rand_index1]['vertices'][rand_coordinate]
      individual.genome[rand_index1]['color'], individual.genome[rand_index2]['color'] = individual.genome[rand_index2]['color'], individual.genome[rand_index1]['color']

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



monalisa = EvolutionaryAlgorithm(
    fitness_function = inverse_image_difference,
    initial_population_function = random_polygon_combinations,
    parent_selection_function = truncation_selection,
    survivor_selection_function = random_selection,
    cross_over_function = random_length_crossover,
    mutation_operator = random_two_gene_swap,
    population_size=100,
    mutation_rate=0.9
)

monalisa.run()
