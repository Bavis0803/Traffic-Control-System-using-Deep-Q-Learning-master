from __future__ import absolute_import, print_function

import csv
import os
import sys
import numpy as np
import torch
import traci
from sumolib import checkBinary
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from your existing files
from sumoClass import SumoClass
from tlsClass import DQN, TLSClass  # Assuming your DQN class is in a file called DQN.py

def test_model(model_path, num_test_runs=5, max_steps=3000):
    # Initialize metrics collection
    all_metrics = {
        'queue_length': [],
        'waiting_time': [],
        'co2_emission': [],
        'average_speed': [],
        'reward': []  # Add reward to metrics
    }
    
    episode_metrics = defaultdict(list)
    
    # Initialize SUMO and TLS objects
    nodes = 4
    edges = 4
    lanes = 3
    
    # Define weights and max_values for reward calculation
    weights = {
        'average_speed': 0.4,   # Reward for high speed
        'co2_emission': -0.15,  # Penalty for high CO2
        'waiting_time': -0.4,   # Penalty for long waiting time
        'queue_length': -0.45   # Penalty for long queue
    }

    max_values = {
        'average_speed': 50.0,  # km/h
        'co2_emission': 1000.0, # g/s
        'waiting_time': 1000.0, # seconds
        'queue_length': 200.0   # number of vehicles
    }
    
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()
    
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    tlsObject = TLSClass()
    # Load trained model
    try:
        tlsObject.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        # Set epsilon to 0 for deterministic behavior during testing
        tlsObject.epsilon = 0
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run test simulations
    for episode in range(num_test_runs):
        print(f"Starting test episode {episode + 1}/{num_test_runs}")
        step_metrics = {
            'queue_length': [],
            'waiting_time': [],
            'co2_emission': [],
            'average_speed': [],
            'reward': []  # Add reward to step metrics
        }
        
        try:
            # Start SUMO - ensure we're starting a fresh instance for each episode
            sumoObject.generate_routefile()  # Generate route file before starting TraCI
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", 
                         f"test_tripinfo_{episode}.xml", "--quit-on-end"])
            
            # Initialize traffic light
            traci.trafficlight.setPhase("0", 1)
            traci.trafficlight.setPhaseDuration("0", 200)
            
            step = 0
            while step < max_steps:
                # Check if simulation is still running before proceeding
                if traci.simulation.getMinExpectedNumber() <= 0:
                    print(f"Episode {episode+1} finished: No more vehicles in simulation")
                    break
                    
                traci.simulationStep()
                
                # Get state
                state = sumoObject.get_state()
                
                # Debug info
                if step % 100 == 0:
                    print(f"Episode {episode+1}, Step {step}: State shapes = {[s.shape for s in state]}")
                
                # Get action from model
                action = tlsObject.act(state)
                
                # Calculate metrics
                vehicles = traci.vehicle.getIDList()
                
                # Queue length
                queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
                step_metrics['queue_length'].append(queue_length)
                
                # Waiting time
                waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
                step_metrics['waiting_time'].append(waiting_time)
                
                # CO2 emission (mg/s to g/s)
                co2_emission = sum(traci.vehicle.getCO2Emission(veh) for veh in vehicles) / 1000  
                step_metrics['co2_emission'].append(co2_emission)
                
                # Average speed (m/s to km/h)
                if vehicles:
                    average_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles) / len(vehicles) * 3.6
                else:
                    average_speed = 0
                step_metrics['average_speed'].append(average_speed)
                
                # Calculate reward using the same formula as in training
                norm_average_speed = average_speed / max_values['average_speed']
                norm_co2_emission = co2_emission / max_values['co2_emission']
                norm_waiting_time = waiting_time / max_values['waiting_time'] 
                norm_queue_length = queue_length / max_values['queue_length']

                reward = (weights['average_speed'] * norm_average_speed +
                         weights['co2_emission'] * norm_co2_emission +
                         weights['waiting_time'] * norm_waiting_time +
                         weights['queue_length'] * norm_queue_length)
                
                step_metrics['reward'].append(reward)
                
                # Print reward at regular intervals
                if step % 100 == 0:
                    print(f"Episode {episode+1}, Step {step}: Reward = {reward:.4f}")
                
                # Apply action
                light = np.argmax(state[2].reshape(4,))
                apply_action(action, light, traci)
                
                step += 1
                
        except traci.exceptions.FatalTraCIError as e:
            print(f"SUMO connection error in episode {episode+1}: {e}")
        except Exception as e:
            print(f"Error in episode {episode+1}: {e}")
        finally:
            # Always ensure the connection is closed before starting a new episode
            try:
                if traci.isLoaded():
                    traci.close()
                    print(f"Episode {episode+1} - TraCI connection closed properly")
            except:
                print(f"Episode {episode+1} - TraCI connection already closed or failed to close")
        
        # Calculate episode averages
        for metric, values in step_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                episode_metrics[metric].append(avg_value)
                print(f"Episode {episode+1} - Average {metric}: {avg_value:.4f}")
            else:
                print(f"Episode {episode+1} - No data collected for {metric}")
                episode_metrics[metric].append(0)  # Add zero if no data
    
    # Calculate overall averages
    print("\n=== Overall Test Results ===")
    for metric, values in episode_metrics.items():
        if values:
            overall_avg = sum(values) / len(values)
            all_metrics[metric] = values
            print(f"Overall average {metric}: {overall_avg:.4f}")
    
    # Plot results with reward included
    plot_results(all_metrics, num_test_runs)
    
    # Save results to CSV
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

def plot_results(metrics, num_episodes):
    """
    Plot test results
    """
    plt.figure(figsize=(12, 10))
    
    for i, (metric, values) in enumerate(metrics.items(), 1):
        if values:  # Only plot if we have data
            plt.subplot(2, 2, i)
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
        num_episodes = len(next(iter(metrics.values())))
        for i in range(num_episodes):
            row = {'episode': i + 1}
            for metric, values in metrics.items():
                if i < len(values):
                    row[metric] = values[i]
                else:
                    row[metric] = 0  # Default value if episode data is missing
            writer.writerow(row)
    
    print(f"Results saved to '{filename}'")

if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = 'trained_kernel_initialiser_2.pth'
    
    # Run the test
    test_model(model_path, num_test_runs=5, max_steps=3000)