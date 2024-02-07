

from types import FunctionType
import numpy as np
import copy
import random


# CBS


# inverse distance to be set and then seen as what needs to be achieved


class chormosome():
    def _init_(self, comb: list):
        self.indi = comb[:]

        self.distance = 0
        self.chorm_distance(self.indi) # allowing a new distance value to be generated


    def city_distance(self, x: tuple, y: tuple):
    # x and y are two 2D points each not 2 coordinates of one 2D point.
        return ((x[0]-y[0])*2 + (x[1]-y[1])2)*0.5
    
    
    def get_distance(self):
        return self.distance
    

    def chorm_distance(self, indi: list):
        for i in range(len(indi)-1):
            self.distance += self.city_distance(city_dict[indi[i]], city_dict[indi[i+1]])
            
    
    # mutation algorithm to be changed rn
    def mutation(self, mutation_rate:int):
        rand_numb = random.randint(0,1)
        if rand_numb <= mutation_rate:
            # mutation defined by reversing orders for now and to be changed
            i = random.randint(0,193)
            j = random.randint(0,193)
            self.indi[i],self.indi[j] = self.indi[j], self.indi[i]



         
class EvolAlgo():

    # returning a nested list for all the shuffled city and their possible combination
    # def random_intercity_paths(self, population_size: int):
    #     population = []

    #     for i in range(population_size):
    #         population.append(copy.deepcopy(city_list))
    #         np.random.shuffle(population[i])
        
    #     return population
    

    def _init_(self,  population_size: int):
        self.population = []
        self.population_size = population_size

        # temp variable to store all the city combinations
        temp_pop = []

        # copying only 10 city element right now
        for i in range(population_size):    
            temp_pop.append(copy.deepcopy(city_list))
            np.random.shuffle(temp_pop[i])
            choromo1 = chormosome(temp_pop[i])
            # individual chromosome to be added in the list
            # print(choromo1.indi)
            self.population.append(choromo1)
        
        print("the list is ", self.population[0].indi)

    def crossover(self, chrom1, chrom2, mut_rate):
        # scheme to be implemented using crossover scheme where a randomly selected set of parent 1 would displace the randomly selected set of parent 2

        start = random.randint(50,75)
        end = random.randint(76, 120)
        # start = random.randint(1, 4)
        # end = random.randint(5, 8)
        parent1 = chrom1.indi
        parent2 = chrom2.indi

        offspring1 = [None] * 194
        offspring2 = [None] * 194

        offspring1[start:end+1] = parent1[start:end+1]
        offspring2[start:end+1] = parent2[start:end+1]
        # print(offspring1)
        # print(offspring2)
        # print(end)
        pointer = end + 1
        parent1_pointer = end + 1
        parent2_pointer = end + 1
        # print(start,end, parent2_pointer)
        counter = 0

        while pointer != start:
            # print(pointer, parent2[parent2_pointer])   
            if parent2[parent2_pointer] not in offspring1:    
                offspring1[pointer] = parent2[parent2_pointer]
                pointer += 1
            parent2_pointer += 1
            if parent2_pointer == len(offspring1):
                parent2_pointer = 0
            if pointer == len(offspring1):
                pointer = 0
            # counter+=1

        pointer = end + 1

        while pointer != start:
            # print(pointer, parent2[parent2_pointer])
            if parent1[parent1_pointer] not in offspring2: 
                offspring2[pointer] = parent1[parent1_pointer]
                pointer += 1
            parent1_pointer += 1
            if parent1_pointer == len(offspring2):
                parent1_pointer = 0
            if pointer == len(offspring2):
                pointer = 0

        offspring1 = chormosome(offspring1)
        offspring2 = chormosome(offspring2)
        offspring1.mutation(mut_rate)
        offspring2.mutation(mut_rate)

        return [offspring1, offspring2] 

    def total_fitness(self):
        total = 0
        for i in self.population:
            total += i.get_distance()
        return total
    
    def best_fitness(self):
        best = self.population[0].get_distance()
        for i in self.population[1:]:
            if i.get_distance() < best:
                best = i.get_distance()
        return best
    
    def avg_fitnness(self):
        total = self.total_fitness()
        return total / self.population_size
    
    def random_selection(self, size):
        result = []
        for i in range(size):
            rand_num = random.randint(0, self.population_size - 1)
            result.append(self.population[rand_num])
        # print(result)
        return result

    def truncation_selection(self, size):
        result = []
        result = copy.deepcopy(self.population)
        result.sort(key=lambda k : k.get_distance())
        # print(result)
        return result[:size]
    
    def binary_tournament_selection(self, size):
        result= []
        for i in range(size):
            ind1, ind2 = random.sample(self.population, 2)
            selected = ind1 if ind1.get_distance() < ind2.get_distance() else ind2
            # print(ind1.get_distance(), ind2.get_distance())
            result.append(selected)
            # print(selected.get_distance())
        return result
    
    # fitness propotional to be changed according actual implementation

    def fitness_selection(self, size):
        fitness = []
        total_fitness = self.total_fitness()
        # print(total_fitness)

        # fitness_prob = []  
        normalized_range =[]
        result = []

        for i in self.population:
            fitness.append(i.get_distance())
        
        normalized_range.append(fitness[0]/total_fitness)

        for i in range(1,len(fitness)):
            normalized_value = fitness[i]/total_fitness
            normalized_range.append(normalized_range[i-1] + normalized_value)

        for i in range(size):
            random_numb = random.random()

            for i in range(len(normalized_range)):
                if random_numb < i:
                    result.append(self.population[i])
                    break

        # print(normalized_range)
        return result

    def rank_selection(self, size):
        rank = sorted(self.population, key=lambda x: x.get_distance())
        total_rank = (len(rank) * (len(rank)  + 1)) / 2
        # total_rank  = (10*11)/2
        # print(total)
        # fitness_prob = []  
        normalized_range =[]
        result = []
        
        normalized_range.append(1/total_rank)

        for i in range(1,len(rank)):
            normalized_value = (i+1)/total_rank
            # print(rank[i].get_distance(), end = ', ')
            # print()
            normalized_range.append(normalized_range[i-1] + normalized_value)
        

        for i in range(size):
            random_numb = random.random()

            for i in range(len(normalized_range)):
                if random_numb < i:
                    result.append(rank[i])
                    break
        
        # print(normalized_range)
        return result


    def run_generation(self,parent_sel, survivor_sel, popul, offsp, mr, num_gen):
        parents = []
        for i in range(num_gen):
            if parent_sel == 1:
                parents = self.random_selection(offsp)
            elif parent_sel == 2:
                parents = self.binary_tournament_selection(offsp)
            elif parent_sel == 3:
                parents = self.fitness_selection(offsp)
            elif parent_sel == 4:
                parents = self.rank_selection(offsp)
            elif parent_sel == 5:
                parents = self.truncation_selection(offsp)
            
            for j in range(0,offsp,2):
                selected_parents = random.sample(parents, 2)
                self.population += self.crossover(selected_parents[0], selected_parents[1], mr)
            print("the total populatin size is ", len(self.population))
            if survivor_sel == 1:
                self.population = self.random_selection(popul)
            elif survivor_sel == 2:
                self.population = self.binary_tournament_selection(popul)
            elif survivor_sel == 3:
                self.population = self.fitness_selection(popul)
            elif survivor_sel == 4:
                self.population = self.rank_selection(popul)
            elif survivor_sel == 5:
                self.population = self.truncation_selection(popul)

            print("the best fit for the ", i, " generation is: ", self.best_fitness())

        
myalgo = EvolAlgo(300)
a = (myalgo.population[0])
# print("list after printing is", a.indi)
# result = myalgo.crossover((myalgo.population[0]), (myalgo.population[1]),0.5)

# result = myalgo.binary_tournament_selection()

myalgo.run_generation(5,5,300,150,0.8,500)