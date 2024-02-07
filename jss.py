from types import FunctionType
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm

def calculate_schedule_time(job_schedule: list) -> float:
    pass

class JobSchedule(Individual):
    def __init__(self, genome):
        fitness = calculate_schedule_time(genome)
        super().__init__(genome, fitness)

    def mutate() -> None:
        pass