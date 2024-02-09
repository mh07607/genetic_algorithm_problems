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

max_polygons_per_image = 50
min_polygons_per_image = 5

# Load the reference image
reference_image = np.array(Image.open('data/monalisa.png'))


def draw_polygon(draw, color, vertices) -> None:
    draw.polygon(vertices, fill=color)


def generate_random_polygon():
    num_vertices = random.randint(3, 10)  # Random number of vertices (3 to 6)
    vertices = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_vertices)]
    # color = color_pallete[np.random.randint(0, 5)]  # Random RGB color
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(10,60))  # Random RGB color
    return {'vertices': vertices, 'color': color}


def image_difference(genome):
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    # Draw each polygon
    for polygon in genome:
        draw_polygon(draw, polygon['color'], polygon['vertices'])
    # Calculate the absolute pixel-wise difference
    diff = np.abs(reference_image - np.array(image))
    # Calculate the mean difference as the fitness score
    image_difference = np.mean(diff)
    return image_difference


class PolygonImage(Individual):
    def __init__(self, genome):
        fitness = image_difference(genome)
        super().__init__(genome, fitness)
        self.mutation_rates = {"add_polygon": 0.125,
                               "remove_polygon": 0.125,
                               "move_polygon": 0.125,
                               "add_point": 0.05,
                               "remove_point": 0.05,
                               "large_point_change": 0.05,
                               "medium_point_change": 0.05,
                               "small_point_change": 0.05,
                               "mutate_red": 0.05,
                               "mutate_green": 0.05,
                               "mutate_blue": 0.05,
                               "mutate_alpha": 0.05}
        
        self.min_points_per_polygon = 3
        self.max_points_per_polygon = 10
        self.mutation_ranges = {"min_points_per_polygon": 3,
                                "max_points_per_polygon": 10}
    
    def add_polygon(self) -> None:
        if(len(self.genome) < max_polygons_per_image):
            index = random.randint(0, len(self.genome)-1)
            vertices = []
            for i in range(self.mutation_ranges["min_points_per_polygon"]):
                vertices.append((random.randint(0, width), random.randint(0, height)))
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(10,60))  # Random RGB color
            self.genome.insert(index, {'vertices': vertices, 'color': color})


    def remove_polygon(self) -> None:
        if(len(self.genome) > min_polygons_per_image):
            index = random.randint(0, len(self.genome)-1)
            del self.genome[index]
            
    def move_polygon(self) -> None:
        index1 = random.randint(0, len(self.genome)-1)
        index2 = random.randint(0, len(self.genome)-1)
        self.genome[index1], self.genome[index2] = self.genome[index2], self.genome[index1]

    def add_point(self, vertices) -> List:
        if(len(vertices > self.max_points_per_polygon)):
            return vertices
        index = random.randint(0, len(vertices)-2)
        prevX, prevY = vertices[index]
        nextX, nextY = vertices[index+1]
        newX, newY = (prevX+nextX)/2, (prevY+nextY)/2
        vertices.append((newX, newY))
        return vertices

    def remove_point(self, vertices) -> List:
        if(len(vertices) < self.min_points_per_polygon):
            return vertices
        index = random.randint(0, len(vertices)-1)
        del vertices[index]
        return vertices

    def mutate_color(self, polygon) -> tuple:
        pass

    def mutate_polygons(self) -> None:
        for i in range(len(self.genome)):
            if(np.random.uniform() <= self.mutation_rates["add_point"]):
                self.genome[i]["vertices"] = self.add_point(self.genome[i]["vertices"])
            if(np.random.uniform() <= self.mutation_rates["remove_point"]):
                self.genome[i]["vertices"] = self.remove_point(self.genome[i]["vertices"])



    def mutate(self) -> None:
        # Add polygon
        if(np.random.uniform() <= self.mutation_rates["add_polygon"]):
            self.add_polygon()
        if(np.random.uniform() <= self.mutation_rates["remove_polygon"]):
            self.remove_polygon()
        if(np.random.uniform() <= self.mutation_rates["move_polygon"]):
            self.move_polygon()

        self.mutate_polygons()
        
        self.fitness = image_difference(self.genome)



    def save(self, image_name):
        image = Image.new("RGB", (width, height), color=(0,0,0))
        draw = ImageDraw.Draw(image, "RGBA")
        # Draw each polygon
        for polygon in self.genome:
            draw_polygon(draw, polygon['color'], polygon['vertices'])
        image.save(image_name)


def random_polygon_combinations(population_size: int) -> List[PolygonImage]:
    population = []
    for i in range(population_size):
        genome = [generate_random_polygon() for _ in range(min_polygons_per_image)]
        population.append(PolygonImage(genome))
    return population


def random_length_crossover(parent1: PolygonImage, parent2: PolygonImage) -> tuple:
    length1 = len(parent1.genome)
    length2 = len(parent1.genome)
    length = max(length1, length2)

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
        parent2_pointer = (parent2_pointer + 1) % length

    pointer = 0

    while None in offspring2:
        #if parent1.genome[parent1_pointer] not in offspring2:
        offspring2[pointer] = parent1.genome[parent1_pointer]
        pointer += 1
        parent1_pointer = (parent1_pointer + 1) % length

    offspring1 = PolygonImage(offspring1)
    offspring2 = PolygonImage(offspring2)

    return offspring1, offspring2


class MonaLisa_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def run(self, num_iterations: int=10, num_generations: int=10000):
      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation()
          if(i % 500 == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("\nAverage fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
            best_individual.save("data/fake_monalisa/fake_monalisa_"+str(j)+"_"+str(i)+".png")

        self.population = self.initial_population_function(self.population_size)
        

monalisa = MonaLisa_EvolutionaryAlgorithm(
    initial_population_function = random_polygon_combinations,
    parent_selection_function = 'truncation',
    survivor_selection_function = 'random',
    cross_over_function = random_length_crossover,
    population_size = 100,
    mutation_rate = 0.5,
    num_offsprings=100
)
monalisa.run(num_generations=10000)
