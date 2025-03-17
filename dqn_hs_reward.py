from __future__ import absolute_import, print_function

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import traci
from sumolib import checkBinary

# Import from your existing files
from sumoClass import SumoClass
from tlsClass import DQN, TLSClass

# Define reward weights
weights = {
    'average_speed': 0.4,
    'co2_emission': -0.15,
    'waiting_time': -0.35,
    'queue_length': -0.45
}

def test_model(model_path, num_test_runs=5, max_steps=10000):
    # Initialize metrics collection
    all_metrics = {
        'queue_length': [],
        'waiting_time': [],
        'co2_emission': [],
        'average_speed': [],
        'reward': []
    }
    
    episode_metrics = defaultdict(list)
    raw_rewards = []  # Store raw rewards for normalization
    
    # Initialize SUMO and TLS objects
    nodes, edges, lanes = 4, 4, 3
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()
    
    sumoBinary = checkBinary('sumo-gui') if not options.nogui else checkBinary('sumo')
    
    tlsObject = TLSClass()

    # Load trained model
    try:
        tlsObject.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        tlsObject.epsilon = 0  # No exploration during testing
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run test simulations
    for episode in range(num_test_runs):
        print(f"\nStarting test episode {episode + 1}/{num_test_runs}")

        step_metrics = {
            'queue_length': [],
            'waiting_time': [],
            'co2_emission': [],
            'average_speed': [],
            'reward': []
        }
        
        try:
            sumoObject.generate_routefile()
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", 
                         f"test_tripinfo_{episode}.xml", "--quit-on-end"])
            
            traci.trafficlight.setPhase("0", 1)
            traci.trafficlight.setPhaseDuration("0", 200)
            
            step = 0
            last_phase = traci.trafficlight.getPhase("0")

            while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                
                # Get state
                state_irl = sumoObject.get_state_irl()
                state = sumoObject.get_state()
                
                # Get action from model
                action = tlsObject.act(state)

                # Collect traffic metrics
                vehicles = traci.vehicle.getIDList()
                queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
                waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
                co2_emission = sum(traci.vehicle.getCO2Emission(veh) for veh in vehicles) / 1000  
                average_speed = (sum(traci.vehicle.getSpeed(veh) for veh in vehicles) / len(vehicles) * 3.6) if vehicles else 0

                # Compute raw reward
                raw_reward = (weights['average_speed'] * average_speed +
                              weights['co2_emission'] * co2_emission +
                              weights['waiting_time'] * waiting_time +
                              weights['queue_length'] * queue_length)
                
                step_metrics['queue_length'].append(queue_length)
                step_metrics['waiting_time'].append(waiting_time)
                step_metrics['co2_emission'].append(co2_emission)
                step_metrics['average_speed'].append(average_speed)
                step_metrics['reward'].append(raw_reward)
                raw_rewards.append(raw_reward)

                # Apply action
                light = np.argmax(state[2].reshape(4,))
                apply_action(action, light, traci)

                step += 1
            
        except traci.exceptions.FatalTraCIError as e:
            print(f"SUMO connection error in episode {episode+1}: {e}")
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
        finally:
            if traci.isLoaded():
                traci.close()
                print(f"Episode {episode+1} - TraCI connection closed properly")
        
        # Store episode averages
        for metric, values in step_metrics.items():
            avg_value = sum(values) / len(values) if values else 0
            episode_metrics[metric].append(avg_value)
            print(f"Episode {episode+1} - Average {metric}: {avg_value:.2f}")

    # Normalize rewards using robust scaling
    min_bound = np.percentile(raw_rewards, 5)
    max_bound = np.percentile(raw_rewards, 95)
    normalized_rewards = [(max(0, min(10, (r - min_bound) / (max_bound - min_bound) * 10))) for r in raw_rewards]
    episode_metrics["reward"] = normalized_rewards

    # Print overall test results
    print("\n=== Overall Test Results ===")
    for metric, values in episode_metrics.items():
        if values:
            overall_avg = sum(values) / len(values)
            all_metrics[metric] = values
            print(f"Overall average {metric}: {overall_avg:.2f}")

    # Save and plot results
    plot_results(all_metrics, num_test_runs)
    save_results_to_csv(episode_metrics, "test_results.csv")

def apply_action(action, light, traci):
    """
    Apply the selected action to the traffic light
    """
    try:
        if action == 0:
            if light == 0:
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return  # Exit if simulation is done
                    traci.trafficlight.setPhase('0', 0)
                    traci.simulationStep()
            elif light == 1:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
            elif light == 2:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
            elif light == 3:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
        
        elif action == 1:
            if light == 1:
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
            elif light == 0:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
            elif light == 2:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
            elif light == 3:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
        
        elif action == 2:
            if light == 0 or light == 1:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
        
        elif action == 3:
            if light == 0 or light == 1:
                for i in range(2):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    traci.simulationStep()
    except traci.exceptions.FatalTraCIError:
        # Catch the FatalTraCIError and let the main loop handle it
        print("TraCI connection closed during action execution")
        return
0
def plot_results(metrics, num_episodes):
    """
    Plot test results
    """
    plt.figure(figsize=(12, 10))
    
    for i, (metric, values) in enumerate(metrics.items(), 1):
        if values:
            plt.subplot(2, 3, i)
            plt.plot(range(1, len(values) + 1), values, 'o-', label=metric)
            plt.title(f'Average {metric.replace("_", " ").title()} per Episode')
            plt.xlabel('Episode')
            plt.ylabel(metric.replace("_", " ").title())
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    print("Results plot saved as 'test_results.png'")

def save_results_to_csv(metrics, filename):
    """
    Save test results to CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['episode'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(next(iter(metrics.values())))):
            row = {'episode': i + 1}
            for metric, values in metrics.items():
                row[metric] = values[i] if i < len(values) else 0
            writer.writerow(row)
    
    print(f"Results saved to '{filename}'")

if __name__ == "__main__":
    model_path = 'trained_kernel_initialiser_2.pth'
    test_model(model_path, num_test_runs=5, max_steps=10000)
