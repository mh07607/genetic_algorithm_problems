from PIL import Image, ImageDraw
import imageio
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm

# Set up the screen
width, height = 1200, 1200
polygons_per_image = 50

# Load the reference image
reference_image = np.array(Image.open('data/monalisa.png').convert('RGBA'))

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
    # color = color_pallete[np.random.randint(0, 5)]  # Random RGB color
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255), np.random.randint(10,60))  # Random RGB color
    return {'vertices': vertices, 'color': color}


def image_difference(genome):
    image = Image.new("RGBA", (width, height), color=(0, 0, 0))
    overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    # Draw each polygon
    for polygon in genome:
        draw = ImageDraw.Draw(overlay)
        draw_polygon(draw, polygon['color'], polygon['vertices'])
        image = Image.alpha_composite(image, overlay)
    # Capture the current screen as an image
    # Calculate the absolute pixel-wise difference
    diff = np.abs(reference_image - np.array(image))
    # Calculate the mean difference as the fitness score
    image_difference = np.mean(diff)
    return image_difference


class PolygonImage(Individual):
    def __init__(self, genome):
        fitness = image_difference(genome)
        super().__init__(genome, fitness)
    

    def mutate(self) -> None:
        rand_index1 = random.randint(0, len(self.genome)-1)
        rand_index2 = random.randint(0, len(self.genome)-1)
        if(np.random.uniform() <= 0.7):
            rand_coordinate = random.randint(0, min(len(self.genome[rand_index1]['vertices']), len(self.genome[rand_index2]['vertices']))-1)
            self.genome[rand_index1]['vertices'][rand_coordinate], self.genome[rand_index2]['vertices'][rand_coordinate] = self.genome[rand_index2]['vertices'][rand_coordinate], self.genome[rand_index1]['vertices'][rand_coordinate]
        else:
        # individual.genome[rand_index1]['color'], individual.genome[rand_index2]['color'] = individual.genome[rand_index2]['color'], individual.genome[rand_index1]['color']
            color1 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(10,60))
            color2 = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(10,60))
            self.genome[rand_index1]['color'], self.genome[rand_index2]['color'] = color1, color2


    def save(self, image_name):
        image = Image.new("RGBA", (width, height), color=(0,0,0))
        overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
        # Draw each polygon
        for polygon in self.genome:
            draw = ImageDraw.Draw(overlay)
            draw_polygon(draw, polygon['color'], polygon['vertices'])
            image = Image.alpha_composite(image, overlay)
        
        image.save(image_name)


def random_polygon_combinations(population_size: int) -> List[PolygonImage]:
    population = []
    for i in range(population_size):
        genome = [generate_random_polygon() for _ in range(polygons_per_image)]
        population.append(PolygonImage(genome))
    return population


def random_length_crossover(parent1: Individual, parent2: Individual) -> tuple:
    length = len(parent1.genome)
    
    start = random.randint(0, int(length-3))
    end = random.randint(start, int(length-2))

    offspring1 = [None] * length
    offspring2 = [None] * length

    offspring1[start:end+1] = parent1.genome[start:end+1]
    offspring2[start:end+1] = parent2.genome[start:end+1]

    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1

    while None in offspring1:
        #if parent2.genome[parent2_pointer] not in offspring1:
        offspring1[pointer % length] = parent2.genome[parent2_pointer]
        pointer += 1
        parent2_pointer = (parent2_pointer + 1) % len(parent2.genome)

    pointer = 0

    while None in offspring2:
        #if parent1.genome[parent1_pointer] not in offspring2:
        offspring2[pointer] = parent1.genome[parent1_pointer]
        pointer += 1
        parent1_pointer = (parent1_pointer + 1) % len(parent1.genome)

    offspring1 = PolygonImage(offspring1)
    offspring2 = PolygonImage(offspring2)

    return offspring1, offspring2


class MonaLisa_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def run(self, num_iterations: int=10, num_generations: int=10000):
      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation()
          if(i % 100 == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("Average fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
            best_individual.save("fake_monalisa_"+str(j)+"_"+str(i)+".png")

        self.population = self.initial_population_function()
        

monalisa = MonaLisa_EvolutionaryAlgorithm(
    initial_population_function = random_polygon_combinations,
    parent_selection_function = 'truncation',
    survivor_selection_function = 'random',
    cross_over_function = random_length_crossover,
    population_size = 100,
    mutation_rate = 0.5,
    num_offsprings=50
)
monalisa.run(num_generations=10000)
