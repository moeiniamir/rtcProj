# import json
# import random
# # import matplotlib.pyplot as plt

# class Task:
#     def __init__(self, id, arrival_time, deadline, execution_time):
#         self.id = id
#         self.arrival_time = arrival_time
#         self.deadline = deadline
#         self.execution_time = execution_time
#         self.start_time = 0
#         self.finish_time = 0
#         self.waiting_time = 0
#         self.response_time = 0
#         self.slack_time = 0

# class Scheduler:
#     def __init__(self, num_tasks, mutation_rate, crossover_rate, max_iter):
#         self.num_tasks = num_tasks
#         self.mutation_rate = mutation_rate
#         self.population = []
#         self.best_fitness = 0
#         self.best_chromosome = []

#     def initialize(self):
#         # code to initialize population with random chromosomes
#         pass

#     def crossover(self):
#         # code to perform crossover between parent chromosomes
#         pass

#     def mutate(self):
#         # code to perform mutation on child chromosomes
#         pass

#     def fitness(self):
#         # code to calculate fitness of each chromosome in the population
#         pass

#     def selection(self):
#         # code to perform selection of fittest chromosomes for next generation
#         pass

#     def evolve(self, num_generations):
#         # code to run GA algorithm for given number of generations
#         pass

#     def run(self, tasks):
#         # code to run the scheduler for given tasks using GA algorithm
#         pass

# class GA_Scheduler(Scheduler):
#     """
#     A class representing the GA algorithm for task scheduling.
#     """

#     def __init__(self, num_tasks, mutation_rate, crossover_rate, max_iter):
        
#         self.pop_size = num_tasks
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.max_iter = max_iter
#         super().__init__(num_tasks, mutation_rate, crossover_rate, max_iter)

#     def initialize_population(self, num_tasks):
#         """
#         Initializes the population with random task orders.

#         Args:
#             num_tasks (int): Number of tasks.

#         Returns:
#             list: A list of task orders (chromosomes).
#         """
#         population = []
#         for i in range(self.pop_size):
#             chromosome = [j for j in range(num_tasks)]
#             random.shuffle(chromosome)
#             population.append(chromosome)
#         return population

#     def calculate_fitness(self, population, tasks):
#         """
#         Calculates the fitness of each chromosome in the population.

#         Args:
#             population (list): A list of task orders (chromosomes).
#             tasks (list): A list of Task objects.

#         Returns:
#             list: A list of tuples containing the chromosome and its fitness.
#         """
#         fitness_scores = []
#         for chromosome in population:
#             full_time, final_time, wait_time, resp_time, slack_time = self.evaluate(chromosome, tasks)
#             # print(self.evaluate(chromosome, tasks))
#             fitness = 1 / (1 + slack_time)
#             fitness_scores.append((chromosome, fitness))
#         return fitness_scores

#     def evaluate(self, chromosome, tasks):
#         """
#         Evaluates a chromosome by simulating the execution of the tasks.

#         Args:
#             chromosome (list): A list representing the order of tasks to be executed.
#             tasks (list): A list of Task objects.

#         Returns:
#             tuple: A tuple containing the full time, final time, wait time, response time, and slack time.
#         """
#         full_time = 0
#         final_time = 0
#         wait_time = 0
#         resp_time = 0
#         slack_time = 0
#         for i, task_id in enumerate(chromosome):
#             task = tasks[task_id]
#             if i == 0:
#                 task.start_time = task.arrival_time
#             else:
#                 prev_task = tasks[chromosome[i-1]]
#                 task.start_time = max(prev_task.finish_time, task.arrival_time)
#             task.finish_time = task.start_time + task.execution_time
#             task.waiting_time = task.start_time - task.arrival_time
#             task.response_time = task.finish_time - task.arrival_time
#             if task.finish_time > task.deadline:
#                 slack_time += task.deadline - task.finish_time
#             else:
#                 slack_time += task.deadline - task.finish_time
#             full_time = max(full_time, task.finish_time)
#             final_time = task.finish_time
#             wait_time += task.waiting_time
#             resp_time += task.response_time
#         return full_time, final_time, wait_time, resp_time, slack_time

#     def selection(self, fitness_scores):
#         """
#         Selects two parent chromosomes using tournament selection.

#         Args:
#             fitness_scores (list): A list of tuples containing the chromosome and its fitness.

#         Returns:
#             tuple: A tuple containing the two parent chromosomes.
#         """
#         tournament_size = int(len(fitness_scores) * 1)
#         tournament = random.sample(fitness_scores, tournament_size)
#         tournament.sort(key=lambda x: x[1], reverse=True)

#         # print(tournament)
#         parent1 = tournament[0][0]
#         parent2 = tournament[1][0]
#         return parent1, parent2

#     def crossover(self, parent1, parent2):
#         """
#         Performs single point crossover between two parent chromosomes.

#         Args:
#             parent1 (list): The first parent chromosome.
#             parent2 (list): The second parent chromosome.

#         Returns:
#             list: The child chromosome.
#         """
#         crossover_point = random.randint(1, len(parent1)-1)
#         child = parent1[:crossover_point] + parent2[crossover_point:]
#         return child

#     def mutate(self, chromosome):
#         """
#         Performs mutation on a chromosome by swapping two random tasks.

#         Args:
#             chromosome (list): The chromosome to be mutated.

#         Returns:
#             list: The mutated chromosome.
#         """
#         if random.random() < self.mutation_rate:
#             idx1 = random.randint(0, len(chromosome)-1)
#             idx2 = random.randint(0, len(chromosome)-1)
#             chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
#         return chromosome

#     def evolve(self, num_generations, tasks):
#         """
#         Evolves the population using GA algorithm for given number of generations.

#         Args:
#             num_generations (int): Number of generations to run the algorithm for.
#             tasks (list): A list of Task objects.

#         Returns:
#             tuple: A tuple containing the best chromosome and its fitness.
#         """
#         self.population = self.initialize_population(self.num_tasks)
#         for gen in range(num_generations):
#             fitness_scores = self.calculate_fitness(self.population, tasks)
#             fittest_chromosome, fittest_fitness = max(fitness_scores, key=lambda x: x[1])
#             if fittest_fitness > self.best_fitness:
#                 self.best_fitness = fittest_fitness
#                 self.best_chromosome = fittest_chromosome
#             new_population = []
#             while len(new_population) < self.pop_size:
#                 parent1, parent2 = self.selection(fitness_scores)
#                 child = self.crossover(parent1, parent2)
#                 child = self.mutate(child)
#                 new_population.append(child)
#             self.population = new_population
#         return self.best_chromosome, self.best_fitness

#     def run(self, tasks):
#         """
#         Runs the scheduler for given tasks using GA algorithm.

#         Args:
#             tasks (list): A list of Task objects.

#         Returns:
#             tuple: A tuple containing the best chromosome and its fitness.
#         """
#         best_chromosome, best_fitness = self.evolve(self.max_iter, tasks)
#         for i, task_id in enumerate(best_chromosome):
#             task = tasks[task_id]
#             if i == 0:
#                 task.start_time = task.arrival_time
#             else:
#                 prev_task = tasks[best_chromosome[i-1]]
#                 task.start_time = max(prev_task.finish_time, task.arrival_time)
#             task.finish_time = task.start_time + task.execution_time
#             task.waiting_time = task.start_time - task.arrival_time
#             task.response_time = task.finish_time - task.arrival_time
#             if task.finish_time > task.deadline:
#                 task.slack_time = task.deadline - task.finish_time
#             else:
#                 task.slack_time = task.deadline - task.finish_time
#         return best_chromosome, best_fitness

# def generate_tasks(num_tasks):
#     """
#     Generates random tasks.

#     Args:
#         num_tasks (int): Number of tasks to generate.

#     Returns:
#         list: A list of Task objects.
#     """
#     tasks = []
#     for i in range(num_tasks):
#         id = i+1
#         # arrival_time = random.randint(0, 10)
#         # deadline = random.randint(arrival_time+1, 15)
#         # execution_time = random.randint(1, 5)

#         arrival_time = random.randint(0, 10)
#         deadline = 15
#         execution_time = 2
#         task = Task(id, arrival_time, deadline, execution_time)
#         tasks.append(task)
#     return tasks

# if __name__ == '__main__':
#     # Generate random tasks
#     num_tasks = 5
#     tasks = generate_tasks(num_tasks)

#     # Run GA algorithm to schedule tasks
#     ga_scheduler = GA_Scheduler(num_tasks=num_tasks, mutation_rate=0.1, crossover_rate=0.8, max_iter=100)
#     best_chromosome, best_fitness = ga_scheduler.run(tasks)

#     # Print results
#     for i in tasks:
#         print('id',i.id)
#         print('arrival_time',i.arrival_time)
#         print('deadline',i.deadline)
#         print('slack', i.slack_time)
#         # print('execution_time',i.execution_time)
#     print("Best chromosome:", best_chromosome)
#     print("Best fitness:", best_fitness)





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
        
        self.pop_size = num_tasks
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
            fitness = 1 / (np.exp(-1 * slack_time))
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
        task_copy.sort(key=lambda x: x.arrival_time)

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
            slack_time += max(0, task.slack_time)

        return full_time, final_time, wait_time, resp_time, slack_time

    def selection(self, fitness_scores):
        """
        Selects two parent chromosomes using tournament selection.

        Args:
            fitness_scores (list): A list of tuples containing the chromosome and its fitness.

        Returns:
            tuple: A tuple containing the two parent chromosomes.
        """
        candidates = random.sample(fitness_scores, 2)
        parent1 = max(candidates, key=lambda x: x[1])[0]

        candidates = random.sample(fitness_scores, 2)
        parent2 = max(candidates, key=lambda x: x[1])[0]

        return parent1, parent2

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
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
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

        best_chromosome, best_fitness = max(fitness_scores, key=lambda x: x[1])
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

def plot_results(num_tasks_list, ga_results, improved_ga_results):
    fig, ax = plt.subplots()
    x = range(len(num_tasks_list))
    ax.bar(x, ga_results, width=0.3, label="GA")
    ax.bar([i + 0.3 for i in x], improved_ga_results, width=0.3, label="Improved GA")
    ax.set_xticks([i + 0.15 for i in x])
    ax.set_xticklabels(num_tasks_list)
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Average Time")
    ax.set_title("Average Times for Different Number of Tasks")
    ax.legend()

    plt.savefig("results.png")
    plt.show()


def main():
    num_tasks_list = [10,]
    ga_results = []
    improved_ga_results = []

    for num_tasks in num_tasks_list:
        # tasks = generate_tasks(num_tasks)
        tasks = [Task(id = 0, arrival_time = 37, deadline = 72, execution_time = 9),
                    Task(id = 1, arrival_time = 25, deadline = 62, execution_time = 8),
                    Task(id = 2, arrival_time = 43, deadline = 96, execution_time = 4),
                    Task(id = 3, arrival_time = 3, deadline = 53, execution_time = 1),
                    Task(id = 4, arrival_time = 44, deadline = 55, execution_time = 7),
                    Task(id = 5, arrival_time = 21, deadline = 66, execution_time = 2),
                    Task(id = 6, arrival_time = 0, deadline = 68, execution_time = 4),
                    Task(id = 7, arrival_time = 39, deadline = 65, execution_time = 9),
                    Task(id = 8, arrival_time = 15, deadline = 18, execution_time = 9),
                    Task(id = 9, arrival_time = 40, deadline = 92, execution_time = 5)]

        


        ga = GA_Scheduler(num_tasks, mutation_rate=0.01, crossover_rate=0.95, max_iter=10000)
        final_time, slack_time = ga.run(tasks)
        ga_results.append(final_time)

        for task in tasks:
            print(f'deadline = {task.deadline}, slack_time = {task.slack_time}), finish_time = {task.finish_time}')

    print(ga_results)

if __name__ == "__main__":
    main()