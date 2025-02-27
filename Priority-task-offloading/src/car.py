import random
from src.task import Task

class Car:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.local_ram = random.randint(8, 16)  # Added local RAM for the car
        self.local_rom = random.randint(50, 100)  # Added local ROM for the car
        self.tasks = []

    def generate_tasks(self, num_tasks):
        for i in range(num_tasks):
            task = Task(f"Task_{self.id}_{i}", self.x, self.y, random.randint(1, 8), random.randint(10, 50), random.randint(1, 5))
            self.tasks.append(task)

    def can_execute_task_locally(self, task):
        return self.local_ram >= task.ram and self.local_rom >= task.rom

    def execute_task_locally(self, task):
        if self.can_execute_task_locally(task):
            self.local_ram -= task.ram
            self.local_rom -= task.rom
            print(f"    Task {task.id} executed locally on Car {self.id}")
            return True
        return False

