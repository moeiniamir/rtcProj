import csv
import numpy as np
import random

def generate_random_periods_discrete(num_periods: int, num_sets: int, available_periods: list):
    """
    Generate a matrix of (num_sets x num_periods) random periods chosen randomly in the list of available periods.
    Args:
        - num_periods: The number of periods in a period set.
        - num_sets: Number of sets to generate.
        - available_periods: A list of available periods.
    """
    try:
        periods = np.random.choice(available_periods, size=(num_sets, num_periods)).tolist()
    except AttributeError:
        p = np.array(available_periods)
        periods = p[np.random.randint(len(p), size=(num_sets, num_periods))].tolist()

    with open('discrete_periods.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i, period_set in enumerate(periods):
            writer.writerow([f'Period Set {i+1}'] + period_set)

    return periods

def generate_uunifastdiscard(nsets: int, u: float, n: int, filename: str):
    """
    The UUniFast algorithm was proposed by Bini for generating task
    utilizations on uniprocessor architectures.
    The UUniFast-Discard algorithm extends it to multiprocessor by
    discarding task sets containing any utilization that exceeds 1.
    This algorithm is easy and widely used. However, it suffers from very
    long computation times when n is close to u. Stafford's algorithm is
    faster.
    Args:
        -   n  : The number of tasks in a task set.
        -   u  : Total utilization of the task set.
        -   nsets  : Number of sets to generate.
        -   filename  : Name of the CSV file to save the results.
    Returns   nsets   of   n   task utilizations.
    """
    sets = []
    while len(sets) < nsets:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(sumU)
        
        if all(ut <= 1 for ut in utilizations):
            sets.append(utilizations)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Task ' + str(i) for i in range(1, n+1)])
        for utilizations in sets:
            writer.writerow(utilizations)

    return sets
    
    
def generate_tasksets(utilizations, periods, filename):
    """
    Take a list of task utilization sets and a list of task period sets and
    return a list of couples (c, p) sets. The computation times are truncated
    at a precision of 10^-10 to avoid floating point precision errors.
    Args:
        -   utilizations : The list of task utilization sets. 
        -   periods : The list of task period sets.
        -   filename : The name of the CSV file to save the results in.
    Returns:
        For the above example, it returns:
            [[(30.0, 100), (20.0, 50), (800.0, 1000)],
             [(20.0, 200), (450.0, 500), (5.0, 10)]]
    """
    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    result = [[(trunc(ui * pi, 6), trunc(pi, 6)) for ui, pi in zip(us, ps)]
              for us, ps in zip(utilizations, periods)]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)

    return result
