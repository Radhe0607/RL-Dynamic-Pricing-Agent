import random

def generate_demand():
    return random.randint(50, 150)

if __name__ == "__main__":
    print(generate_demand())