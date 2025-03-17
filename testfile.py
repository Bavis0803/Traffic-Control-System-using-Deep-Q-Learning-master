# test_performance.py

import csv

import numpy as np
import torch
import traci
from sumolib import checkBinary

from sumoClass import SumoClass
from tlsClass import TLSClass


def test_performance(trained_model_path, output_csv="actions_dqn.csv", num_episodes=10):
    nodes = 4
    edges = 4
    lanes = 3

    sumoObject = SumoClass(nodes, edges, lanes)
    sumoBinary = checkBinary('sumo-gui') if not sumoObject.get_options().nogui else checkBinary('sumo')

    # Load the trained model
    tlsObject = TLSClass()
    tlsObject.load(trained_model_path)

    # Open CSV file to save actions
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "action"])  # Header

        # Initialize performance metrics
        total_reward = 0
        total_average_speed = 0
        total_waiting_time = 0
        total_queue_length = 0
        total_co2_emission = 0

        for episode in range(num_episodes):
            print(f"Testing episode {episode + 1}/{num_episodes}")
            stepz = 0
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml", "--start", "--quit-on-end"])
            
            while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:
                traci.simulationStep()
                try:
                    state = sumoObject.get_state()
                except ValueError as e:
                    print(f"Error in get_state: {e}")
                    continue  # Skip this step if error occurs

                action = tlsObject.act(state)

                # Save action to CSV
                writer.writerow([stepz, action])

                # Collect performance metrics
                vehicles = traci.vehicle.getIDList()
            # Tính tốc độ trung bình (km/h)
                average_speed = sum(traci.vehicle.getSpeed(v) for v in vehicles) / max(len(vehicles), 1) * 3.6
                
                # Tính tổng lượng khí thải CO2 (kg/s)
                co2_emission = sum(traci.vehicle.getCO2Emission(v) for v in vehicles) / 1000
                
                # Đếm số xe đang chờ (tốc độ < 0.1 m/s)
                waiting_vehicles = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
                
                # Tính độ dài hàng đợi (tổng số xe dừng trên tất cả các cạnh)
                queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
                total_average_speed += average_speed
                total_co2_emission += co2_emission
                total_waiting_time += waiting_vehicles
                total_queue_length += queue_length

                # Reward calculation
                total_reward += calculate_reward(average_speed, waiting_vehicles, queue_length, co2_emission)

                stepz += 1

            traci.close()

    # Calculate average performance metrics
    avg_reward = total_reward / num_episodes
    avg_waiting_time = total_waiting_time / num_episodes
    avg_queue_length = total_queue_length / num_episodes
    avg_co2_emission = total_co2_emission / num_episodes    

    print(f"✅ Average Reward: {avg_reward}")
    print(f"✅ Average Waiting Time: {avg_waiting_time}")
    print(f"✅ Average Queue Length: {avg_queue_length}")
    print(f"✅ Average CO2 Emission: {avg_co2_emission}")
    print(f"✅ Hành động đã lưu vào file: {output_csv}")

def calculate_reward(average_speed, waiting_time, queue_length, co2_emission):
    """Define a reward function based on performance metrics"""
    return 0.4 * average_speed - 0.35 * waiting_time - 0.45 * queue_length - 0.15 * co2_emission


if __name__ == "__main__":
    trained_model_path = 'trained_kernel_initialiser_2.pth'  # Path to trained model
    test_performance(trained_model_path, output_csv="actions_dqn.csv", num_episodes=10)
