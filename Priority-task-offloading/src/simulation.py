#main------------------------------------------------

import random
import pandas as pd
from src.car import Car
from src.server import Server
from src.task import Task
from src.q_learning import QLearning
from src.utils import calculate_distance
import heapq
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        self.cars = []
        self.servers = []
        self.graph = {}  # Adjacency list for the graph
        self.q_learning = QLearning(num_states=10000, num_actions=10)  # Increased state space
        self.successful_offloads = 0
        self.failed_offloads = 0
        self.local_executions = 0  # Counter for tasks executed locally
        self.server_utilization = {}
        self.task_distribution = {}  # Track tasks offloaded to each server
        self.distances = [] 


    def create_cars(self, num_cars):
        for i in range(num_cars):
            car = Car(i, random.randint(0, 100), random.randint(0, 100))
            self.cars.append(car)

    def create_servers(self, num_servers):
        for i in range(num_servers):
            server = Server(i, random.randint(0, 100), random.randint(0, 100), random.randint(8, 24), random.randint(50, 150))
            self.servers.append(server)
            print(f"  Server {server.id} is at location ({server.x}, {server.y}) "
                f"with RAM: {server.ram}, ROM: {server.rom}")
        
        # Initialize the server utilization and task distribution
        self.server_utilization = {server.id: {'ram': 0, 'rom': 0} for server in self.servers}
        self.task_distribution = {server.id: 0 for server in self.servers}  # Track tasks per server
        
        self.create_graph()

    def create_graph(self):
        nodes = self.cars + self.servers
        for node1 in nodes:
            self.graph[node1.id] = []
            for node2 in nodes:
                if node1 != node2:
                    distance = calculate_distance(node1.x, node1.y, node2.x, node2.y)
                    self.graph[node1.id].append((node2.id, distance))

    def dijkstra(self, start_id):
        distances = {node: float('inf') for node in self.graph}
        distances[start_id] = 0
        priority_queue = [(0, start_id)]  # (distance, node_id)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

    def load_tasks(self, file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            car_id = row['car_id']
            task = Task(row['task_id'], row['car_x'], row['car_y'], row['task_ram'], row['task_rom'], row['priority'])
            self.cars[car_id].tasks.append(task)


    def offload_tasks(self):
        for car in self.cars:
            print(f"\nCar {car.id} at location ({car.x}, {car.y}) with resources (RAM: {car.local_ram}, ROM: {car.local_rom}):")
            for task in car.tasks:
                print(f"\n  Task {task.id} (RAM: {task.ram}, ROM: {task.rom}, Priority: {task.priority})")

                # First, try to execute the task locally on the car
                if car.execute_task_locally(task):
                    self.local_executions += 1
                    continue  # Move to the next task
                else:
                    print(f"Task {task.id} could not be executed locally on Car {car.id}. Attempting to offload.")

                # If the task can't be executed locally, offload it to the servers
                state = self.get_state(car, task)
                action = self.q_learning.choose_action(state)
                server = self.servers[action]

                if server.can_handle_task(task):
                    print(f"    Attempting to offload to Server {server.id} "
                        f"(Available RAM: {server.available_ram}, Available ROM: {server.available_rom})")
                    if server.available_ram >= task.ram and server.available_rom >= task.rom:
                        server.available_ram -= task.ram
                        server.available_rom -= task.rom
                        self.successful_offloads += 1
                        print(f"    Successfully offloaded to Server {server.id} "
                            f"(Remaining RAM: {server.available_ram}, Remaining ROM: {server.available_rom})")
                        self.task_distribution[server.id] += 1
                    else:
                        print(f"    Not enough resources on Server {server.id} to offload task.")
                        self.failed_offloads += 1
                else:
                    # Try to find the nearest server that can handle the task
                    nearest_server = self.find_nearest_server(car, task)
                    if nearest_server:
                        print(f"    Server {server.id} cannot handle the task. "
                            f"Trying nearest Server {nearest_server.id} "
                            f"(Available RAM: {nearest_server.available_ram}, Available ROM: {nearest_server.available_rom})")
                        if nearest_server.available_ram >= task.ram and nearest_server.available_rom >= task.rom:
                            nearest_server.available_ram -= task.ram
                            nearest_server.available_rom -= task.rom
                            self.successful_offloads += 1
                            print(f"    Successfully offloaded to nearest Server {nearest_server.id} "
                                f"(Remaining RAM: {nearest_server.available_ram}, Remaining ROM: {nearest_server.available_rom})")
                            self.task_distribution[nearest_server.id] += 1
                        else:
                            print(f"    Not enough resources on nearest Server {nearest_server.id} to offload task.")
                            self.failed_offloads += 1
                    else:
                        print(f"    Could not offload to any server - Insufficient resources")
                        self.failed_offloads += 1

                # Track distance covered during offloading
                distance_to_server = calculate_distance(car.x, car.y, server.x, server.y)
                self.distances.append(distance_to_server)

                next_state = self.get_state(car, task)
                reward = 1 if server.can_handle_task(task) else -1  # Reward or penalty
                self.q_learning.update_q_table(state, action, reward, next_state)

    def find_nearest_server(self, car, task):
        nearest_server = None
        min_distance = float('inf')
    
        for server in self.servers:
            # Only consider servers that can handle the task
            if server.can_handle_task(task):
                distance = calculate_distance(car.x, car.y, server.x, server.y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_server = server
    
        return nearest_server

    def get_state(self, car, task):
        state = (car.x + car.y + task.ram + task.rom) % 100
        for server in self.servers:
            state = (state + server.available_ram + server.available_rom + calculate_distance(car.x, car.y, server.x, server.y)) % 10000
        return int(state)  # Ensure state is an integer



    def plot_results(self):
        plt.figure(figsize=(6, 4))
        # Update the bar chart to include tasks executed locally
        plt.bar(["Locally Executed", "Successful Offloads", "Failed Offloads"], 
                [self.local_executions, self.successful_offloads, self.failed_offloads], 
                color=['blue', 'green', 'red'])
        plt.title("Task Offloading Success Rate")
        plt.ylabel("Number of Tasks")
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.bar(self.task_distribution.keys(), self.task_distribution.values(), color='purple')
        plt.title("Task Distribution Across Servers")
        plt.ylabel("Number of Tasks")
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(self.distances, color='cyan')
        plt.title("Distance Covered in Task Offloading")
        plt.xlabel("Task Number")
        plt.ylabel("Distance (units)")
        plt.show()


    def plot_locations(self):
        car_x = [car.x for car in self.cars]
        car_y = [car.y for car in self.cars]
        server_x = [server.x for server in self.servers]
        server_y = [server.y for server in self.servers]

        plt.figure(figsize=(10, 10))
        plt.scatter(car_x, car_y, color='blue', label='Cars', marker='o')
        plt.scatter(server_x, server_y, color='red', label='Servers', marker='x')
        plt.title("Car and Server Locations")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def worst_offload_tasks(self):
        """
        Implements the 'worst' algorithm by randomly selecting a server for each task,
        regardless of its resources or distance.
        """
        successful_offloads = 0
        failed_offloads = 0
        distances = []

        for car in self.cars:
            for task in car.tasks:
                random_server = random.choice(self.servers)

                if random_server.can_handle_task(task):
                    random_server.available_ram -= task.ram
                    random_server.available_rom -= task.rom
                    successful_offloads += 1
                else:
                    failed_offloads += 1

                # Track the distance for comparison
                distance_to_server = calculate_distance(car.x, car.y, random_server.x, random_server.y)
                distances.append(distance_to_server)

        return {
            "successful": successful_offloads,
            "failed": failed_offloads,
            "distances": distances
        }

    def plot_comparison(self, original_results, worst_results):
        """
        Compares the performance of the current algorithm with the worst algorithm
        using a bar chart.
        """
        metrics = ['Successful Offloads', 'Failed Offloads']
        original_values = [original_results['successful_offloads'], original_results['failed_offloads']]
        worst_values = [worst_results['successful'], worst_results['failed']]

        x = range(len(metrics))
        plt.bar(x, original_values, width=0.4, label='Original Algorithm', align='center')
        plt.bar([i + 0.4 for i in x], worst_values, width=0.4, label='Worst Algorithm', align='center')

        plt.xticks([i + 0.2 for i in x], metrics)
        plt.ylabel('Count')
        plt.title('Comparison of Task Offloading Algorithms')
        plt.legend()
        plt.show()

        # Plot distance comparison
        plt.figure(figsize=(6, 4))
        plt.plot(original_results['distances'], label="Original Algorithm", color='green')
        plt.plot(worst_results['distances'], label="Worst Algorithm", color='red', linestyle='--')
        plt.title("Distance Comparison Between Algorithms")
        plt.xlabel("Task Number")
        plt.ylabel("Distance (units)")
        plt.legend()
        plt.show()

    def run_simulation(self):
        """
        Orchestrates the simulation for both the original and worst algorithms
        and plots the results.
        """
        print("Running the simulation with the original algorithm...")
        original_results = {
            "successful_offloads": self.successful_offloads,
            "failed_offloads": self.failed_offloads,
            "distances": self.distances
        }

        print("Running the simulation with the worst algorithm...")
        worst_results = self.worst_offload_tasks()

        print("Plotting the comparison results...")
        self.plot_comparison(original_results, worst_results)
#mainend----------------------------------


