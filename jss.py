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

class JobSchedule(Individual):
    def __init__(self, genome):
        fitness = calculate_schedule_time(genome)
        super().__init__(genome, fitness)


    def mutate(self) -> None:
        rand_index1 = random.randint(0, len(self.genome)-1)
        rand_index2 = random.randint(0, len(self.genome)-1)

        self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]


def random_job_steps(population_size: int) -> List(JobSchedule):
    pass

def calculate_schedule_time(job_schedule: list) -> float:
    pass




class JSS_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def run(self):
        pass