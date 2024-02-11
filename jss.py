from types import FunctionType
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.colors as plc

''' Loading dataset '''

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

''' Decide which dataset to test here '''
datasets = read_dataset(dataset_path=dataset_path)
current_dataset = datasets['abz7']

# Deciding colors for 
# colors = {}
# for i in range(current_dataset["num_jobs"]):
#     color = (np.random.uniform(), np.random.uniform(), np.random.uniform())
#     colors["Job-"+str(i)] = color
tasks = ['Job-'+str(i) for i in range(current_dataset["num_jobs"])]
# Generate a list of distinct colors for the tasks
pallette = plc.qualitative.Alphabet
# Create a dictionary to map tasks to colors
colors = dict(zip(tasks, pallette))



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
    job_schedule = copy.copy(job_schedule)
    jobs = copy.deepcopy(current_dataset["jobs"])
    num_jobs = len(jobs)
    while jobs != [None] * num_jobs:
        if(None in machines_dict.values()):
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

    def save_as_gantt_chart(self, name: str) -> None:
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

        machines_timeline = []
        job_schedule = copy.copy(self.genome)
        jobs = copy.deepcopy(current_dataset["jobs"])
        num_jobs = len(jobs)
        while jobs != [None] * num_jobs:
            if(None in machines_dict.values()):
                for index, job_number in enumerate(job_schedule):
                    if not job_currently_in_any_machine(job_number, machines_dict):
                        machine_number = jobs[job_number][0][0]
                        if(machines_dict[machine_number] == None):
                            machine_number, time_step = jobs[job_number].pop()
                            # saving time_step twice so that I can use it to get time taken
                            machines_dict[machine_number] = [job_number, time_step, time_step]
                            del job_schedule[index]
                            if(jobs[job_number] == []):
                                jobs[job_number] = None

            for i in machines_dict:
                if(machines_dict[i] != None):
                    machines_dict[i][1] = machines_dict[i][1]-1
                    if(machines_dict[i][1] <= 0):
                        machines_timeline.append(
                            dict(Task="Machine-"+str(i),
                                  Start=total_time-machines_dict[i][2],
                                  Finish=total_time, 
                                  Resource="Job-"+str(machines_dict[i][0]))
                                )
                        machines_dict[i] = None
            
            total_time += 1
        
        ''' Plotting '''
        # we have defined colors above
        fig = ff.create_gantt(sorted(machines_timeline, key=lambda x: x["Task"]), 
                              colors=colors, index_col='Resource', show_colorbar=True,
                      group_tasks=True, bar_width=0.1)
        fig.update_layout(xaxis_type='linear')
        fig.update_layout(height=800)
        fig.write_image(name)


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
    def run(self, num_iterations: int=10, num_generations: int=10000) -> tuple:
      best_fitnesses = [[] for _ in range(num_iterations)]
      average_fitnesses = [[] for _ in range(num_iterations)]
      x_offset = num_generations // 5

      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation()
          if(i % x_offset == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("\nAverage fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
            # print(best_individual.genome)
            best_fitnesses[j].append(best_individual.fitness)
            average_fitnesses[j].append(average_fitness)


      self.population = self.initial_population_function(self.population_size)
      return best_individual, best_fitnesses, average_fitnesses

'''Test run'''
# jss = JSS_EvolutionaryAlgorithm(
#     initial_population_function = random_job_steps,
#     parent_selection_function = 'truncation',
#     survivor_selection_function = 'random',
#     cross_over_function = JSS_random_length_crossover,
#     population_size = 100,
#     mutation_rate = 0.5,
#     num_offsprings=10
# )
# jss.run(num_generations=1000)

''' Generating Graphs '''
selection_pairs = [
                    ('fitness', 'random', 100, 0.5, 50),
                    ('binary', 'truncation', 100, 0.9, 20), 
                    ('truncation', 'truncation', 100, 0.9, 50), 
                    ('random', 'random', 100, 0.5, 10),
                    ('fitness', 'rank', 100, 0.5, 50),
                    ('truncation', 'random', 100, 0.5, 50),
                    ('rank', 'binary', 100, 0.5, 100),
                  ]

num_generations = 500
num_iterations = 5
x_offset = num_generations // 5

for parent_selection, survivor_selection, population_size, mutation_rate, num_offsprings in selection_pairs:
  print(parent_selection, survivor_selection)
  jss = JSS_EvolutionaryAlgorithm(
      initial_population_function = random_job_steps,
      parent_selection_function = parent_selection,
      survivor_selection_function = survivor_selection,
      cross_over_function = JSS_random_length_crossover,
      population_size = population_size,
      mutation_rate = mutation_rate,
      num_offsprings=num_offsprings
  )

  best_individual, best_fitnesses, average_fitnesses = jss.run(num_generations=num_generations, num_iterations=num_iterations)
  best_fitnesses = np.array(best_fitnesses).T.tolist()
  average_fitnesses = np.array(average_fitnesses).T.tolist()
  x = []
  y1 = []
  y2 = []

  for i in range(len(best_fitnesses)):
    x.append(i * x_offset)
    y1.append(np.average(best_fitnesses[i]))
    y2.append(np.average(average_fitnesses[i]))

  plt.figure()
  plt.plot(x, y1, label='Average best fitness')
  plt.plot(x, y2, label='Average average fitness')

  plt.xlabel('Number of generations')
  plt.ylabel('Average average/best fitness values')
  plt.title(parent_selection + ', ' +survivor_selection + ', ' +
          str(population_size) + ', ' +
          str(mutation_rate) + ', ' +
          str(num_offsprings))
  plt.legend()
  plt.tight_layout()
  plt.savefig('data/jss_analysis/'+parent_selection+'_'+survivor_selection+'.png')  # Save as PNG

  # Plot job schedule of best individual in the last iteration
  best_individual.save_as_gantt_chart('data/jss_analysis/'+parent_selection
                                      +'_'+survivor_selection+'_gantt_chart.png')
  print(best_individual.genome)
