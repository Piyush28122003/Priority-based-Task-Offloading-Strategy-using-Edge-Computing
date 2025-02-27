import pandas as pd
import random
import os

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Define the number of cars and tasks
num_cars = 10
tasks_per_car = 5

# Generate data for tasks
data = []
for car_id in range(num_cars):
    for task_id in range(tasks_per_car):
        task = {
            'car_id': car_id,
            'task_id': f'Task_{car_id}_{task_id}',
            'car_x': random.randint(0, 100),
            'car_y': random.randint(0, 100),
            'task_ram': random.randint(1, 8),
            'task_rom': random.randint(10, 50),
            'priority': random.randint(1, 5)
        }
        data.append(task)

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/task_data.csv', index=False)

print("task_data.csv has been created successfully.")