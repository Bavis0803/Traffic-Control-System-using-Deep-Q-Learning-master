import numpy as np
import torch
import traci
from stable_baselines3 import PPO
from sumolib import checkBinary
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# Import your models and classes
from sumoClass import SumoClass
from tlsClass import TLSClass, DQN
from DeepMAXIRL import RewardNet

def evaluate_both_models(num_episodes=5, max_steps=3000):
    """
    Compare DQN and PPO models using the same test conditions
    """
    # Initialize metrics collection
    metrics = {
        'dqn': {
            'reward': [],
            'queue_length': [],
            'waiting_time': [],
            'co2_emission': [],
            'average_speed': []
        },
        'ppo': {
            'reward': [],
            'queue_length': [],
            'waiting_time': [],
            'co2_emission': [],
            'average_speed': []
        }
    }
    
    # Load models
    dqn_model_path = 'trained_kernel_initialiser_2.pth'
    ppo_model_path = 'ppo_traffic_policy.zip'
    reward_model_path = 'reward_model.pth'
    
    # Test DQN
    print("=== Testing DQN Model ===")
    metrics['dqn'] = test_dqn(dqn_model_path, num_episodes, max_steps)
    
    # Test PPO
    print("\n=== Testing PPO Model ===")
    metrics['ppo'] = test_ppo(ppo_model_path, reward_model_path, num_episodes, max_steps)
    
    # Compare and visualize results
    compare_results(metrics)
    
    return metrics

def test_dqn(model_path, num_test_runs=5, max_steps=3000):
    """
    Test DQN model performance
    """
    # Initialize metrics collection
    episode_metrics = defaultdict(list)
    
    # Initialize SUMO and TLS objects
    nodes, edges, lanes = 4, 4, 3
    
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
        print(f"Successfully loaded DQN model from {model_path}")
        tlsObject.epsilon = 0  # Deterministic behavior for testing
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return defaultdict(list)
    
    # Run test simulations
    for episode in range(num_test_runs):
        print(f"Starting DQN test episode {episode + 1}/{num_test_runs}")
        step_metrics = defaultdict(list)
        
        try:
            # Start SUMO with a common seed for fair comparison
            sumoObject.generate_routefile(seed=episode+100)
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--quit-on-end"])
            
            # Initialize traffic light
            traci.trafficlight.setPhase("0", 1)
            traci.trafficlight.setPhaseDuration("0", 200)
            
            step = 0
            while step < max_steps:
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
                
                traci.simulationStep()
                
                # Get state
                state = sumoObject.get_state()
                
                # Get action from model
                action = tlsObject.act(state)
                
                # Calculate metrics
                vehicles = traci.vehicle.getIDList()
                queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
                waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
                co2_emission = sum(traci.vehicle.getCO2Emission(veh) for veh in vehicles) / 1000  # mg/s to g/s
                
                if vehicles:
                    average_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles) / len(vehicles) * 3.6  # m/s to km/h
                else:
                    average_speed = 0
                
                # Calculate reward (using DQN's formula)
                max_values = {
                    'average_speed': 50.0,  # km/h
                    'co2_emission': 1000.0, # g/s
                    'waiting_time': 1000.0, # seconds
                    'queue_length': 200.0   # number of vehicles
                }
                
                weights = {
                    'average_speed': 0.4,   # Reward for high speed
                    'co2_emission': -0.15,  # Penalty for high CO2
                    'waiting_time': -0.4,   # Penalty for long waiting time
                    'queue_length': -0.45   # Penalty for long queue
                }
                
                norm_average_speed = average_speed / max_values['average_speed']
                norm_co2_emission = co2_emission / max_values['co2_emission']
                norm_waiting_time = waiting_time / max_values['waiting_time'] 
                norm_queue_length = queue_length / max_values['queue_length']

                reward = (weights['average_speed'] * norm_average_speed +
                         weights['co2_emission'] * norm_co2_emission +
                         weights['waiting_time'] * norm_waiting_time +
                         weights['queue_length'] * norm_queue_length)
                
                # Store metrics
                step_metrics['queue_length'].append(queue_length)
                step_metrics['waiting_time'].append(waiting_time)
                step_metrics['co2_emission'].append(co2_emission)
                step_metrics['average_speed'].append(average_speed)
                step_metrics['reward'].append(reward)
                
                # Apply action
                light = np.argmax(state[2].reshape(4,))
                apply_action(action, light, traci)
                
                step += 1
                
        except Exception as e:
            print(f"Error in DQN episode {episode+1}: {e}")
        finally:
            if traci.isLoaded():
                traci.close()
        
        # Calculate episode averages
        for metric, values in step_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                episode_metrics[metric].append(avg_value)
                print(f"DQN Episode {episode+1} - Average {metric}: {avg_value:.4f}")
    
    # Calculate overall averages
    print("\n=== DQN Overall Test Results ===")
    overall_metrics = {}
    for metric, values in episode_metrics.items():
        if values:
            overall_avg = sum(values) / len(values)
            overall_metrics[metric] = overall_avg
            print(f"DQN Overall average {metric}: {overall_avg:.4f}")
    
    return episode_metrics

def test_ppo(model_path, reward_model_path, num_test_runs=5, max_steps=3000):
    """
    Test PPO model performance
    """
    # Initialize metrics collection
    episode_metrics = defaultdict(list)
    
    # Load models
    try:
        reward_model = RewardNet(state_dim=4, action_dim=4)
        reward_model.load_state_dict(torch.load(reward_model_path))
        reward_model.eval()
        
        ppo_model = PPO.load(model_path)
        print(f"Successfully loaded PPO model from {model_path}")
        print(f"Successfully loaded Reward model from {reward_model_path}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return defaultdict(list)
    
    # Initialize SUMO 
    nodes, edges, lanes = 4, 4, 3
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()
    
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    # Run test simulations
    for episode in range(num_test_runs):
        print(f"Starting PPO test episode {episode + 1}/{num_test_runs}")
        step_metrics = defaultdict(list)
        
        try:
            # Start SUMO with the same seed as DQN for fair comparison
            sumoObject.generate_routefile(seed=episode+100)
            traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--quit-on-end"])
            
            # Initialize traffic light
            traci.trafficlight.setPhase("0", 1)
            traci.trafficlight.setPhaseDuration("0", 200)
            
            step = 0
            while step < max_steps:
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
                
                traci.simulationStep()
                
                # Get state
                state_irl = sumoObject.get_state_irl()
                state_values = [
                    state_irl.get("average_speed", 0),
                    state_irl.get("co2_emission", 0),
                    state_irl.get("waiting_vehicles", 0),
                    state_irl.get("queue_length", 0)
                ]
                state = np.array([float(v) for v in state_values], dtype=np.float32)
                
                # Get action from PPO model
                action, _ = ppo_model.predict(state, deterministic=True)
                
                # Calculate metrics
                vehicles = traci.vehicle.getIDList()
                queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
                waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
                co2_emission = sum(traci.vehicle.getCO2Emission(veh) for veh in vehicles) / 1000  # mg/s to g/s
                
                if vehicles:
                    average_speed = sum(traci.vehicle.getSpeed(veh) for veh in vehicles) / len(vehicles) * 3.6  # m/s to km/h
                else:
                    average_speed = 0
                
                # Calculate reward using the reward model
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor([int(action)], dtype=torch.long)
                reward = reward_model(state_tensor, action_tensor).item()
                
                # Store metrics
                step_metrics['queue_length'].append(queue_length)
                step_metrics['waiting_time'].append(waiting_time)
                step_metrics['co2_emission'].append(co2_emission)
                step_metrics['average_speed'].append(average_speed)
                step_metrics['reward'].append(reward)
                
                # Apply action (assuming the same actions as DQN)
                light = 0  # Default value, might need adjustment based on your PPO implementation
                apply_action(int(action), light, traci)
                
                step += 1
                
        except Exception as e:
            print(f"Error in PPO episode {episode+1}: {e}")
        finally:
            if traci.isLoaded():
                traci.close()
        
        # Calculate episode averages
        for metric, values in step_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                episode_metrics[metric].append(avg_value)
                print(f"PPO Episode {episode+1} - Average {metric}: {avg_value:.4f}")
    
    # Calculate overall averages
    print("\n=== PPO Overall Test Results ===")
    overall_metrics = {}
    for metric, values in episode_metrics.items():
        if values:
            overall_avg = sum(values) / len(values)
            overall_metrics[metric] = overall_avg
            print(f"PPO Overall average {metric}: {overall_avg:.4f}")
    
    return episode_metrics

def apply_action(action, light, traci):
    """
    Apply the selected action to the traffic light
    """
    try:
        if action == 0:
            if light == 0:
                for i in range(30):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return
                    traci.trafficlight.setPhase('0', 0)
                    traci.simulationStep()
            # ... rest of action implementation
            # (Implementation shortened for brevity but should match your original code)
    except traci.exceptions.FatalTraCIError:
        print("TraCI connection closed during action execution")
        return

def compare_results(metrics):
    """
    Compare and visualize results from both models
    """
    # Prepare data for comparison charts
    metrics_to_plot = ['reward', 'queue_length', 'waiting_time', 'co2_emission', 'average_speed']
    
    plt.figure(figsize=(18, 12))
    
    # Plot each metric comparison
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        
        dqn_values = metrics['dqn'][metric]
        ppo_values = metrics['ppo'][metric]
        
        # Plot only if we have data for both
        if dqn_values and ppo_values:
            x = range(1, len(dqn_values) + 1)
            plt.plot(x, dqn_values, 'bo-', label='DQN')
            plt.plot(x, ppo_values, 'ro-', label='PPO')
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xlabel('Episode')
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
            plt.grid(True)
    
    # Add overall comparison summary
    plt.subplot(2, 3, 6)
    
    # Calculate overall averages for both models
    dqn_avgs = {}
    ppo_avgs = {}
    
    for metric in metrics_to_plot:
        if metrics['dqn'][metric]:
            dqn_avgs[metric] = sum(metrics['dqn'][metric]) / len(metrics['dqn'][metric])
        if metrics['ppo'][metric]:
            ppo_avgs[metric] = sum(metrics['ppo'][metric]) / len(metrics['ppo'][metric])
    
    # Create bar chart for overall comparison
    if dqn_avgs and ppo_avgs:
        labels = [m.replace("_", " ").title() for m in metrics_to_plot]
        dqn_vals = [dqn_avgs.get(m, 0) for m in metrics_to_plot]
        ppo_vals = [ppo_avgs.get(m, 0) for m in metrics_to_plot]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, dqn_vals, width, label='DQN')
        plt.bar(x + width/2, ppo_vals, width, label='PPO')
        plt.title('Overall Performance Comparison')
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Comparison chart saved as 'model_comparison.png'")
    
    # Save comparison data to CSV
    save_comparison_to_csv(metrics, 'model_comparison.csv')

def save_comparison_to_csv(metrics, filename):
    """
    Save comparison results to CSV file
    """
    # Get the number of episodes (assuming both have the same number)
    num_episodes = max(
        max([len(values) for values in metrics['dqn'].values()]),
        max([len(values) for values in metrics['ppo'].values()])
    )
    
    with open(filename, 'w', newline='') as csvfile:
        # Define all metrics for both models
        fieldnames = ['episode']
        for model in ['dqn', 'ppo']:
            for metric in metrics[model].keys():
                fieldnames.append(f"{model}_{metric}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data for each episode
        for i in range(num_episodes):
            row = {'episode': i + 1}
            
            # Add data for both models
            for model in ['dqn', 'ppo']:
                for metric, values in metrics[model].items():
                    if i < len(values):
                        row[f"{model}_{metric}"] = values[i]
                    else:
                        row[f"{model}_{metric}"] = ""
            
            writer.writerow(row)
    
    print(f"Comparison results saved to '{filename}'")

if __name__ == "__main__":
    evaluate_both_models(num_episodes=5, max_steps=3000)