import pygame
from pygame.locals import QUIT
import numpy as np
import imageio

# Initialize Pygame
pygame.init()

# Set up the screen
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Random Polygons')

def draw_polygon(surface, color, vertices):
    pygame.draw.polygon(surface, color, vertices)

def generate_random_polygon():
    num_vertices = np.random.randint(3, 7)  # Random number of vertices (3 to 6)
    vertices = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_vertices)]
    color = tuple(np.random.randint(0, 256, size=3))  # Random RGB color
    return {'vertices': vertices, 'color': color}

def calculate_image_difference(image1, image2):
    # Calculate the absolute pixel-wise difference
    diff = np.abs(image1 - image2)
    # Calculate the mean difference as the fitness score
    fitness_score = np.mean(diff)
    return fitness_score

def main():
    # Generate 50 random polygons
    polygons = [generate_random_polygon() for _ in range(50)]

    # Load the reference image
    reference_image = imageio.imread('path/to/your/reference_image.png')

    # Set up the screen (without creating a window)
    screen = pygame.Surface((width, height))

    # Draw each random polygon
    for polygon in polygons:
        draw_polygon(screen, polygon['color'], polygon['vertices'])

    # Capture the current screen as an image
    current_image = pygame.surfarray.array3d(screen).swapaxes(0, 1)

    # Calculate the fitness score (image difference)
    fitness_score = calculate_image_difference(reference_image, current_image)
    print(f'Fitness Score: {fitness_score}')

    # Save the generated image
    pygame.image.save(screen, 'path/to/your/generated_image.png')

if __name__ == '__main__':
    main()
