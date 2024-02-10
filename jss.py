from types import FunctionType
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm

dataset_path = "data/jss_data.txt"

def read_dataset(dataset_path: str) -> dict:
    content = ""
    with open(dataset_path, 'r') as file:
        content = file.read()
    datasets = content.split("\n\n")
    datasets_dict = {}
    for dataset in datasets:
        lines = dataset.strip().split("\n")
        dataset_name = lines[0].strip()
        num_jobs, num_machines = lines[1].strip().split(' ')
        num_jobs, num_machines = int(num_jobs), int(num_machines)
        jobs = []
        for j in range(2, num_jobs+2):
            job_steps = lines[j].strip().split(" ")
            job_steps = [job_step for job_step in job_steps if job_step != '']
            job = []
            for k in range(0, len(job_steps), 2):
                machine_number, time_required = int(job_steps[k].strip()), int(job_steps[k+1].strip())
                job.append((machine_number, time_required))
            jobs.append(job)

        datasets_dict[dataset_name] = {"num_jobs": num_jobs,
                                        "num_machines": num_machines,
                                        "jobs": jobs}
    return datasets_dict

datasets = read_dataset(dataset_path=dataset_path)
current_dataset = datasets['abz5']


def calculate_schedule_time(job_schedule: list) -> float:
    def job_currently_in_any_machine(job_number: int, machines_dict: dict) -> bool:
        for i in machines_dict.values():
            if(i == None):
                continue
            if(i[0] == job_number):
                return True
        return False


    num_machines = current_dataset["num_machines"]
    machines_dict = {}
    for i in range(num_machines):
        machines_dict[i] = None
    
    total_time = 0

    job_schedule = copy.deepcopy(job_schedule)
    jobs = copy.deepcopy(current_dataset["jobs"])
    num_jobs = len(jobs)
    while jobs != [None] * num_jobs:
        for index, job_number in enumerate(job_schedule):
            if not job_currently_in_any_machine(job_number, machines_dict):
                machine_number = jobs[job_number][0][0]
                if(machines_dict[machine_number] == None):
                    machine_number, time_step = jobs[job_number].pop()
                    machines_dict[machine_number] = [job_number, time_step]
                    del job_schedule[index]
                    if(jobs[job_number] == []):
                        jobs[job_number] = None

        for i in machines_dict:
            if(machines_dict[i] != None):
                machines_dict[i][1] = machines_dict[i][1]-1
                if(machines_dict[i][1] <= 0):
                    machines_dict[i] = None
        
        total_time += 1
    return total_time
                



class JobSchedule(Individual):
    def __init__(self, genome):
        fitness = calculate_schedule_time(genome)
        super().__init__(genome, fitness)


    def mutate(self) -> None:
        rand_index1 = random.randint(0, len(self.genome)-1)
        rand_index2 = random.randint(0, len(self.genome)-1)

        self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
        self.fitness = calculate_schedule_time(self.genome)


def random_job_steps(population_size: int) -> List[JobSchedule]:
    num_steps = current_dataset["num_machines"]
    num_jobs = current_dataset["num_jobs"]
    population = []
    for i in range(population_size):
        genome = []
        for i in range(num_jobs):
            genome.extend([i] * num_steps)
        np.random.shuffle(genome)
        population.append(JobSchedule(genome))
    return population


def JSS_random_length_crossover(parent1: JobSchedule, parent2: JobSchedule):
    length = len(parent1.genome)
    num_steps = current_dataset["num_machines"]
    start = random.randint(1, length-3)
    end = random.randint(start, length-2)

    offspring1 = [None] * length
    offspring2 = [None] * length

    offspring1[start:end+1] = parent1.genome[start:end+1]
    offspring2[start:end+1] = parent2.genome[start:end+1]

    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1

    while None in offspring1:
        if offspring1.count(parent2.genome[parent2_pointer]) < num_steps:
            offspring1[pointer % length] = parent2.genome[parent2_pointer]
            pointer += 1
        parent2_pointer = (parent2_pointer + 1) % length

    pointer = 0

    while None in offspring2:
        if offspring2.count(parent1.genome[parent1_pointer]) < num_steps:
            offspring2[pointer % length] = parent1.genome[parent1_pointer]
            pointer += 1
        parent1_pointer = (parent1_pointer + 1) % length

    offspring1 = JobSchedule(offspring1)
    offspring2 = JobSchedule(offspring2)

    return offspring1, offspring2

class JSS_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def run(self, num_iterations: int=10, num_generations: int=10000):
      x_offset = num_generations // 20

      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation()
          if(i % x_offset    == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("\nAverage fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
            print(best_individual.genome)

        self.population = self.initial_population_function(self.population_size)


jss = JSS_EvolutionaryAlgorithm(
    initial_population_function = random_job_steps,
    parent_selection_function = 'truncation',
    survivor_selection_function = 'random',
    cross_over_function = JSS_random_length_crossover,
    population_size = 100,
    mutation_rate = 0.5,
    num_offsprings=50
)
jss.run(num_generations=1000)
