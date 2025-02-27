from src.simulation import Simulation

def main():
    sim = Simulation()
    sim.create_cars(10)
    sim.create_servers(10)
    sim.load_tasks('data/task_data.csv')
    
    print("Running the simulation...")
    sim.offload_tasks()  # Runs the original algorithm

    print("Locations of cars and servers")
    sim.plot_locations()

    print("Plotting the results for the original algorithm...")
    sim.plot_results()

    print("Running the simulation with the worst algorithm...")
    worst_results = sim.worst_offload_tasks()

    print("Comparing the original and worst algorithms...")
    sim.plot_comparison(
        original_results={
            "successful_offloads": sim.successful_offloads,
            "failed_offloads": sim.failed_offloads,
            "distances": sim.distances
        },
        worst_results=worst_results
    )

if __name__ == "__main__":
    main()
