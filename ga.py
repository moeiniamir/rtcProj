import json
import random
import numpy as np
# import matplotlib.pyplot as plt


class Task:
    def __init__(self, id, arrival_time, deadline, execution_time):
        self.id = id
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.execution_time = execution_time
        self.start_time = 0
        self.finish_time = 0
        self.waiting_time = 0
        self.response_time = 0
        self.slack_time = 0
        self.laxity = 0


class Scheduler:
    def __init__(self, num_tasks, mutation_rate, crossover_rate, max_iter):
        self.num_tasks = num_tasks
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_fitness = 0
        self.best_chromosome = []

    def initialize(self):
        # code to initialize population with random chromosomes
        pass

    def crossover(self):
        # code to perform crossover between parent chromosomes
        pass

    def mutate(self):
        # code to perform mutation on child chromosomes
        pass

    def fitness(self):
        # code to calculate fitness of each chromosome in the population
        pass

    def selection(self):
        # code to perform selection of fittest chromosomes for next generation
        pass

    def evolve(self, num_generations):
        # code to run GA algorithm for given number of generations
        pass

    def run(self, tasks):
        # code to run the scheduler for given tasks using GA algorithm
        pass

class GA_Scheduler(Scheduler):
    """
    A class representing the GA algorithm for task scheduling.
    """

    def __init__(self, num_tasks, mutation_rate, crossover_rate, max_iter):
        
        self.pop_size = num_tasks * 3
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_iter = max_iter
        super().__init__(num_tasks, mutation_rate, crossover_rate, max_iter)

    def initialize_population(self, num_tasks):
        """
        Initializes the population with random task orders.

        Args:
            num_tasks (int): Number of tasks.

        Returns:
            list: A list of task orders (chromosomes).
        """
        population = []
        for i in range(self.pop_size):
            chromosome = [j for j in range(num_tasks)]
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def calculate_fitness(self, population, tasks):
        """
        Calculates the fitness of each chromosome in the population.

        Args:
            population (list): A list of task orders (chromosomes).
            tasks (list): A list of Task objects.

        Returns:
            list: A list of tuples containing the chromosome and its fitness.
        """
        

        fitness_scores = []
        for chromosome in population:
            full_time, final_time, wait_time, resp_time, slack_time = self.evaluate(chromosome, tasks)
            # fitness = 1 / (1 + slack_time)
            # fitness = 100 * slack_time
            fitness = 1 / (np.exp(slack_time*-50 + resp_time + wait_time))
            fitness_scores.append((chromosome, fitness))
        return fitness_scores

    def evaluate(self, chromosome, tasks):
        """
        Evaluates a chromosome by simulating the execution of the tasks.

        Args:
            chromosome (list): A list representing the order of tasks to be executed.
            tasks (list): A list of Task objects.

        Returns:
            tuple: A tuple containing the full time, final time, wait time, response time, and slack time.
        """
        full_time = 0
        final_time = 0
        wait_time = 0
        resp_time = 0
        slack_time = 0


        task_copy = tasks.copy()
        # task_copy.sort(key=lambda x: x.arrival_time)

        s_time = []

        for task_id in chromosome:
            task = task_copy[task_id]
            task.start_time = max(task.arrival_time, final_time)
            task.finish_time = task.start_time + task.execution_time
            task.waiting_time = task.start_time - task.arrival_time
            task.response_time = task.finish_time - task.arrival_time
            task.slack_time = task.deadline - task.finish_time

            full_time += task.execution_time
            final_time = task.finish_time
            wait_time += task.waiting_time
            resp_time += task.response_time
            # slack_time += max(0, task.slack_time)
            s_time.append(task.slack_time)

        return full_time, final_time, wait_time, resp_time, min(s_time)

    def selection(self, fitness_scores):
        """
        Selects two parent chromosomes using tournament selection.

        Args:
            fitness_scores (list): A list of tuples containing the chromosome and its fitness.

        Returns:
            tuple: A tuple containing the two parent chromosomes.
        """
        # candidates = random.sample(fitness_scores, 2)
        
        print(fitness_scores)
        # print(candidates)
        candidates1 = max(fitness_scores, key=lambda x: x[1])
        fitness_scores_copy = fitness_scores.copy()
        fitness_scores_copy.remove(candidates1)
        candidates2 = max(fitness_scores_copy, key=lambda x: x[1])
        for _ in range(len(fitness_scores)):
            if candidates1 == candidates2:
                fitness_scores_copy.remove(candidates2)
            candidates2 = max(fitness_scores_copy, key=lambda x: x[1])
            if len(fitness_scores_copy) < 2:
                break
        # parent1 = max(candidates, key=lambda x: x[1])[0]

        # candidates = random.sample(fitness_scores, 2)
        # parent2 = max(candidates, key=lambda x: x[1])[0]

        print(candidates1[0])
        print(candidates2[0])

        return candidates1[0], candidates2[0]

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent chromosomes.

        Args:
            parent1 (list): The first parent chromosome.
            parent2 (list): The second parent chromosome.

        Returns:
            list: The child chromosome resulting from crossover.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        return child

    
    def pbc_crossover(self, parent1, parent2):
        """
        Perform position-based crossover (PBC) on two parent chromosomes to create a new child chromosome.

        Parameters:
            parent1 (list): The first parent chromosome.
            parent2 (list): The second parent chromosome.

        Returns:
            list: The new child chromosome.
        """
        n = len(parent1)
        child = [-1] * n

        # Select random positions for the crossover points
        positions = sorted(random.sample(range(n), 2))

        # Copy the genetic material from the parents into the child chromosome
        for i in range(positions[0], positions[1]):
            child[i] = parent1[i]

        # Fill in the remaining positions in the child chromosome with genetic material from the second parent
        j = 0
        for i in range(n):
            if not parent2[i] in child:
                while child[j] != -1:
                    j += 1
                child[j] = parent2[i]

        return child

    import random

    def ox_crossover(self, parent1, parent2):
        """
        Perform order crossover (OX) on two parent chromosomes to create a new child chromosome.

        Parameters:
            parent1 (list): The first parent chromosome.
            parent2 (list): The second parent chromosome.

        Returns:
            list: The new child chromosome.
        """
        n = len(parent1)
        child = [-1] * n

        # Select two random positions for the crossover points
        positions = sorted(random.sample(range(n), 2))

        # Copy genetic material from the first parent into the child chromosome between the crossover points
        child[positions[0]:positions[1]] = parent1[positions[0]:positions[1]]

        # Fill in the remaining positions in the child chromosome with genetic material from the second parent
        j = positions[1]
        for i in range(n):
            if not parent2[i] in child:
                if j == n:
                    j = 0
                child[j] = parent2[i]
                j += 1

        return child

    def mutate(self, chromosome):
        """
        Mutates a chromosome.

        Args:
            chromosome (list): The chromosome to mutate.

        Returns:
            list: The mutated chromosome.
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

        return chromosome

    def swap_mutate(self, chromosome):
        """
        Perform swap mutation on a chromosome to create a new mutated chromosome.

        Parameters:
            chromosome (list): The chromosome to mutate.

        Returns:
            list: The new mutated chromosome.
        """
        n = len(chromosome)
        mutated_chromosome = chromosome.copy()
        if random.random() < self.mutation_rate:
            # Select two random positions in the chromosome to swap
            positions = random.sample(range(n), 2)

            # Swap the elements at the selected positions
            mutated_chromosome[positions[0]], mutated_chromosome[positions[1]] = \
                mutated_chromosome[positions[1]], mutated_chromosome[positions[0]]

            # Remove any duplicates in the mutated chromosome
            mutated_chromosome = list(dict.fromkeys(mutated_chromosome))

        return mutated_chromosome

    def evolve(self, population, fitness_scores):
        """
        Evolves the population by selecting parents, performing crossover and mutation, and generating a new population.

        Args:
            population (list): A list of task orders (chromosomes).
            fitness_scores (list): A list of tuples containing the chromosome and its fitness.

        Returns:
            list: A new list of task orders (chromosomes).
        """
        new_population = []

        for _ in range(len(population)):
            parent1, parent2 = self.selection(fitness_scores)
            child = self.ox_crossover(parent1, parent2)
            # child = self.mutate(child)
            new_population.append(child)

        return new_population

    def run(self, tasks):
        """
        Runs the GA

        """
        population = self.initialize_population(self.num_tasks)


        for _ in range(self.max_iter):
            fitness_scores = self.calculate_fitness(population, tasks)
            population = self.evolve(population, fitness_scores)
            # print(population)
            # print(fitness_scores)

        best_chromosome, best_fitness = max(fitness_scores, key=lambda x: x[1])
        # print(best_chromosome)
        _, final_time, _, _, slack_time = self.evaluate(best_chromosome, tasks)

        return final_time, slack_time

def generate_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        arrival_time = random.randint(0, 50)
        deadline = random.randint(arrival_time + 1, 100)  # Ensures deadline > arrival_time
        execution_time = random.randint(1, 10)
        task = Task(i, arrival_time, deadline, execution_time)
        tasks.append(task)

    return tasks

def read_tasks_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    tasks = []
    for task_data in data["tasks"]:
        task = Task(task_data["id"], task_data["arrival_time"], task_data["deadline"], task_data["execution_time"])
        tasks.append(task)

    return tasks

def write_results_to_file(results):
    with open("results.json", "w") as file:
        json.dump({"results": results}, file)

# def plot_results(num_tasks_list, ga_results, improved_ga_results):
#     fig, ax = plt.subplots()
#     x = range(len(num_tasks_list))
#     ax.bar(x, ga_results, width=0.3, label="GA")
#     ax.bar([i + 0.3 for i in x], improved_ga_results, width=0.3, label="Improved GA")
#     ax.set_xticks([i + 0.15 for i in x])
#     ax.set_xticklabels(num_tasks_list)
#     ax.set_xlabel("Number of Tasks")
#     ax.set_ylabel("Average Time")
#     ax.set_title("Average Times for Different Number of Tasks")
#     ax.legend()

#     plt.savefig("results.png")
#     plt.show()


def main():
    num_tasks_list = [5,]
    ga_results = []
    improved_ga_results = []

    for num_tasks in num_tasks_list:
        # tasks = generate_tasks(num_tasks)
        tasks = [Task(id = 0, arrival_time = 37, deadline = 72, execution_time = 9),
                    Task(id = 1, arrival_time = 25, deadline = 62, execution_time = 8),
                    Task(id = 2, arrival_time = 43, deadline = 96, execution_time = 4),
                    Task(id = 3, arrival_time = 3, deadline = 53, execution_time = 1),
                    Task(id = 4, arrival_time = 44, deadline = 55, execution_time = 7),
                    # Task(id = 5, arrival_time = 21, deadline = 66, execution_time = 2),
                    # Task(id = 6, arrival_time = 0, deadline = 68, execution_time = 4),
                    # Task(id = 7, arrival_time = 39, deadline = 65, execution_time = 9),
                    # Task(id = 8, arrival_time = 15, deadline = 18, execution_time = 9),
                    # Task(id = 9, arrival_time = 40, deadline = 92, execution_time = 5)
                ]

        


        ga = GA_Scheduler(num_tasks, mutation_rate=0.1, crossover_rate=0.9, max_iter=1000)
        final_time, slack_time = ga.run(tasks)
        ga_results.append(final_time)

        for task in tasks:
            print(f'id = {task.id}, deadline = {task.deadline}, slack_time = {task.slack_time}), finish_time = {task.finish_time}')

    print(ga_results)

if __name__ == "__main__":
    main()
