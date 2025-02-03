import random
import pandas as pd
import math

# Calculate Euclidean distance
def distance(x,y):
    return math.sqrt((y[1]-x[1])**2+(y[0]-x[0])**2)

# Finding the closest availble driver/rider
def find_closest_available(coord, available):
    min_dist = 100000000
    closest = None
    for avail in available.keys():
        d = distance(coord, available[avail][0])
        if d < min_dist:
            min_dist = d
            closest = avail
    return closest

def calculate_trip(origin, destination):
    return 3 + 2*distance(origin, destination)

def calculate_petrol(origin, destination):
    return 0.2*distance(origin, destination)

def generate_random(distribution, param):
    if distribution == 'exp':
        return random.expovariate(param)
    elif distribution == 'unif':
        return random.uniform(0,param)
    return -1

def generate_location():
    return (random.uniform(0,20),random.uniform(0,20))

def generate_trip_time(origin, destination):
    d = distance(origin, destination)
    return random.uniform(d/25,d/16)

def add_to_list(element, lst):
    if len(lst) == 0:
        return [element]
    l = 0
    r = len(lst) - 1
    time = element[1]
    while l <= r:
        mid = (l+r)//2
        if time < lst[mid][1]:
            r = mid - 1
        else:
            l = mid + 1
    lst.insert(l,element)
    return lst