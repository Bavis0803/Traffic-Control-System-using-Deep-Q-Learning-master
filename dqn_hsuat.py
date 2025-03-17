from __future__ import absolute_import, print_function

import csv
import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import traci
from sumolib import checkBinary

# Import from your existing files
from sumoClass import SumoClass
from tlsClass import (  # Assuming your DQN class is in a file called DQN.py
    DQN, TLSClass)


def test_model(model_path, num_test_runs=5, max_steps=10000):
    # Initialize metrics collection
    all_metrics = {
        'queue_length': [],
        'waiting_time': [],
        'co2_emission': [],
        'average_speed': []
    }
    
    episode_metrics = defaultdict(list)
    
    # Initialize SUMO and TLS objects
    nodes = 4
    edges = 4
    lanes = 3
    
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
            'average_speed': []
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
            start_time = time.time()
            while step < max_steps:
                # Check if simulation is still running before proceeding
                if traci.simulation.getMinExpectedNumber() <= 0:
                    print(f"Episode {episode+1} finished: No more vehicles in simulation")
                    break
                
                # Debug print for traffic light state
                tls_id = "0"  # Your traffic light ID
                current_phase = traci.trafficlight.getPhase(tls_id)
                current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                current_duration = traci.trafficlight.getPhaseDuration(tls_id)
                remaining_duration = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
                
                elapsed_time = time.time() - start_time
                print(f"Step {step} | Time: {elapsed_time:.2f}s | Phase: {current_phase} | duration=\"{current_duration:.1f}\" state=\"{current_state}\" | Remaining: {remaining_duration:.1f}s")
                
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
                
                # Apply action
                light = np.argmax(state[2].reshape(4,))
                print(f"Applying action {action} with light {light}")
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
                print(f"Episode {episode+1} - Average {metric}: {avg_value:.2f}")
            else:
                print(f"Episode {episode+1} - No data collected for {metric}")
                episode_metrics[metric].append(0)  # Add zero if no data
    
    # Calculate overall averages
    print("\n=== Overall Test Results ===")
    for metric, values in episode_metrics.items():
        if values:
            overall_avg = sum(values) / len(values)
            all_metrics[metric] = values
            print(f"Overall average {metric}: {overall_avg:.2f}")
    
    # Plot results
    plot_results(all_metrics, num_test_runs)
    
    # Save results to CSV
    save_results_to_csv(episode_metrics, "test_results.csv")

def apply_action(action, light, traci):
    """
    Apply the selected action to the traffic light, matching the main simulation loop logic.
    
    Parameters:
    - action (int): The action to apply (0 to 4, corresponding to phases 0 to 4).
    - light (int): The current traffic light phase.
    - traci: The TraCI interface object for SUMO.
    """
    try:
        # Debug function to print TLS state after each phase change
        def print_tls_state():
            tls_id = "0"
            phase = traci.trafficlight.getPhase(tls_id)
            state = traci.trafficlight.getRedYellowGreenState(tls_id)
            duration = traci.trafficlight.getPhaseDuration(tls_id)
            print(f"  - TLS changed to phase={phase} | duration=\"{duration:.1f}\" state=\"{state}\"")
        
        ### Action 0: Set to Phase 0 (GGGGgrrrrrGGGGgrrrrr)
        if action == 0:
            if light == 0:  # Already in phase 0
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
            elif light == 1:  # Transition from phase 1 to 0 using yellow phase 7
                for i in range(4):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    print_tls_state()
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
            else:  # light in [2, 3, 4, 5]
                # Transition to phase 0 with appropriate yellow phase
                if light in [2, 3]:
                    for i in range(4):
                        if traci.simulation.getMinExpectedNumber() <= 0:
                            return
                        traci.trafficlight.setPhase('0', 6)
                        print_tls_state()
                        traci.simulationStep()
                elif light in [4, 5]:
                    for i in range(4):
                        if traci.simulation.getMinExpectedNumber() <= 0:
                            return
                        traci.trafficlight.setPhase('0', 7)
                        print_tls_state()
                        traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
        
        ### Action 1: Set to Phase 1 (rrrrrGGGGgrrrrrGGGGg)
        elif action == 1:
            if light == 1:  # Already in phase 1
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
            elif light == 0:  # Transition from phase 0 to 1 using yellow phase 6
                for i in range(4):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    print_tls_state()
                    traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
            else:  # light in [2, 3, 4, 5]
                # Transition to phase 1 with appropriate yellow phase
                if light in [2, 3]:
                    for i in range(4):
                        if traci.simulation.getMinExpectedNumber() <= 0:
                            return
                        traci.trafficlight.setPhase('0', 6)
                        print_tls_state()
                        traci.simulationStep()
                elif light in [4, 5]:
                    for i in range(4):
                        if traci.simulation.getMinExpectedNumber() <= 0:
                            return
                        traci.trafficlight.setPhase('0', 7)
                        print_tls_state()
                        traci.simulationStep()
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 31)
                    print_tls_state()
                    traci.simulationStep()
        
        ### Action 2: Set to Phase 2 (rrGGrGrrrrrrGGrGrrrr)
        elif action == 2:
            if light == 2:  # Already in phase 2
                for i in range(15):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 2)
                    traci.trafficlight.setPhaseDuration("0", 15)
                    print_tls_state()
                    traci.simulationStep()
            else:  # Transition to phase 2 using yellow phase 6
                for i in range(4):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 6)
                    print_tls_state()
                    traci.simulationStep()
                for i in range(15):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 2)
                    traci.trafficlight.setPhaseDuration("0", 15)
                    print_tls_state()
                    traci.simulationStep()
        
        ### Action 3: Set to Phase 3 (GrrrrrrGGrGrrrrrrGGr)
        elif action == 3:
            if light == 3:  # Already in phase 3
                for i in range(15):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 3)
                    traci.trafficlight.setPhaseDuration("0", 15)
                    print_tls_state()
                    traci.simulationStep()
            else:  # Transition to phase 3 using yellow phase 7
                for i in range(4):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    print_tls_state()
                    traci.simulationStep()
                for i in range(15):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 3)
                    traci.trafficlight.setPhaseDuration("0", 15)
                    print_tls_state()
                    traci.simulationStep()
        
        ### Action 4: Set to Phase 4 (GrrrGGrrrrGrrrGGrrrr)
        elif action == 4:
            if light == 4:  # Already in phase 4
                for i in range(10):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 4)
                    traci.trafficlight.setPhaseDuration("0", 10)
                    print_tls_state()
                    traci.simulationStep()
            else:  # Need to transition to phase 4
                # If not in phase 0 or 1, first transition to phase 0
                if light not in [0, 1]:
                    if light in [2, 3]:
                        for i in range(4):
                            if traci.simulation.getMinExpectedNumber() <= 0:
                                return
                            traci.trafficlight.setPhase('0', 7)
                            print_tls_state()
                            traci.simulationStep()
                    elif light in [4, 5]:
                        for i in range(4):
                            if traci.simulation.getMinExpectedNumber() <= 0:
                                return
                            traci.trafficlight.setPhase('0', 6)
                            print_tls_state()
                            traci.simulationStep()
                    for i in range(5):
                        if traci.simulation.getMinExpectedNumber() <= 0:
                            return
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        print_tls_state()
                        traci.simulationStep()
                # Transition from phase 0 (or if already in 0 or 1) to phase 4 using yellow phase 7
                for i in range(4):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 7)
                    print_tls_state()
                    traci.simulationStep()
                for i in range(10):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 4)
                    traci.trafficlight.setPhaseDuration("0", 10)
                    print_tls_state()
                    traci.simulationStep()
    
    except traci.exceptions.FatalTraCIError:
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


def save_tls_log(tls_log, filename="tls_state_log.csv"):
    """
    Save traffic light state log to CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'step', 'time', 'phase', 'state', 'duration', 'remaining']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in tls_log:
            writer.writerow(entry)
    
    print(f"Traffic light state log saved to '{filename}'")


if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = 'trained_kernel_initialiser_2.pth'
    
    # Run the test
    test_model(model_path, num_test_runs=5, max_steps=10000)