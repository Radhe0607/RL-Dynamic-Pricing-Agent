import random

def generate_demand(price):
    base_demand = 100
    demand = max(0, base_demand - price + random.randint(-10, 10))
    return demand