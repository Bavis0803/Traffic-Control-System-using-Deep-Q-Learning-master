from sumoClass import SumoClass
from tlsClass import TLSClass
import traci
import numpy as np
import torch

def test_performance(trained_model_path, num_episodes=100):
    nodes = 4
    edges = 4
    lanes = 3

    sumoObject = SumoClass(nodes, edges, lanes)
    sumoBinary = checkBinary('sumo-gui') if not sumoObject.get_options().nogui else checkBinary('sumo')

    # Load the trained model
    tlsObject = TLSClass()
    tlsObject.load(trained_model_path)

    # Initialize performance metrics
    total_reward = 0
    total_waiting_time = 0
    total_queue_length = 0

    for episode in range(num_episodes):
        print(f"Testing episode {episode + 1}/{num_episodes}")
        stepz = 0
        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml", "--quit-on-end"])
        
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 5000:
            traci.simulationStep()
            state = sumoObject.get_state()
            action = tlsObject.act(state)

            # Collect performance metrics
            waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
            queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
            total_waiting_time += waiting_time
            total_queue_length += queue_length

            # Reward calculation can be based on the action taken
            total_reward += calculate_reward(action, waiting_time, queue_length)

            stepz += 1

        traci.close()

    # Calculate average performance metrics
    avg_reward = total_reward / num_episodes
    avg_waiting_time = total_waiting_time / num_episodes
    avg_queue_length = total_queue_length / num_episodes

    print(f"Average Reward: {avg_reward}")
    print(f"Average Waiting Time: {avg_waiting_time}")
    print(f"Average Queue Length: {avg_queue_length}")

def calculate_reward(action, waiting_time, queue_length):
    # Define a reward function based on the action taken and the performance metrics
    # This is a placeholder; adjust the logic as needed
    return -waiting_time - queue_length

if __name__ == "__main__":
    trained_model_path = 'trained_kernel_initialiser.pth'  # Path to the trained model
    test_performance(trained_model_path)