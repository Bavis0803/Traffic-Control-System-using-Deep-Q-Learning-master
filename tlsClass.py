from __future__ import absolute_import, print_function

import csv
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traci
from sumolib import checkBinary

from sumoClass import SumoClass


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Branch 1: input (1,12,12)
        self.conv1_b1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2_b1 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        
        # Branch 2: same as branch 1
        self.conv1_b2 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2_b2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        
        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(1028, 128)  # 512 + 512 + 4 = 1028
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x1, x2, x3):
        # Process branch 1
        x1 = F.relu(self.conv1_b1(x1))
        x1 = F.relu(self.conv2_b1(x1))
        x1 = torch.flatten(x1, 1)
        
        # Process branch 2
        x2 = F.relu(self.conv1_b2(x2))
        x2 = F.relu(self.conv2_b2(x2))
        x2 = torch.flatten(x2, 1)
        
        # Process branch 3
        x3 = torch.flatten(x3, 1)
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TLSClass:
    def __init__(self, epsilon_start=0.9, epsilon_end=0.01, epsilon_decay_episodes=50):
        self.gamma = 0.95
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start  # Initial exploration rate
        self.current_episode = 0
        self.learning_rate = 0.001
        self.memory = deque(maxlen=4000)
        self.action_size = 8  # problem
        self.model = DQN()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def update_epsilon(self):
        """Update epsilon based on current episode"""
        self.current_episode += 1
        # Linear decay
        if self.current_episode <= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.current_episode / self.epsilon_decay_episodes)
        else:
            self.epsilon = self.epsilon_end
        print(f"Episode {self.current_episode}, Epsilon: {self.epsilon:.4f}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        # Use epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Process state[0]
            state0_array = np.array(state[0])
            print(f"State[0] shape: {state0_array.shape}")
            if state0_array.ndim == 3:
                # Input shape is (H, W, C) → add batch dim
                input1 = torch.tensor(state0_array, dtype=torch.float32) \
                            .unsqueeze(0).permute(0, 3, 1, 2)
            elif state0_array.ndim == 4:
                # Input shape is already (batch, H, W, C)
                input1 = torch.tensor(state0_array, dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            elif state0_array.ndim == 5:
                # Input shape is (batch, X, H, W, C) -- combine extra dims as needed.
                new_shape = (state0_array.shape[0],
                            state0_array.shape[1],
                            state0_array.shape[2],
                            state0_array.shape[3] * state0_array.shape[4])
                input1 = torch.tensor(state0_array.reshape(new_shape), dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            else:
                raise ValueError("Unexpected dimensions for state[0]: " + str(state0_array.shape))
            
            # Process state[1] in a similar manner (assuming similar shape expectations)
            state1_array = np.array(state[1])
            print(f"State[1] shape: {state1_array.shape}")
            if state1_array.ndim == 3:
                input2 = torch.tensor(state1_array, dtype=torch.float32) \
                            .unsqueeze(0).permute(0, 3, 1, 2)
            elif state1_array.ndim == 4:
                input2 = torch.tensor(state1_array, dtype=torch.float32) \
                            .permute(0, 3, 1, 2)
            else:
                raise ValueError("Unexpected dimensions for state[1]: " + str(state1_array.shape))
            
            # Process state[2] assuming it is a flat vector with 4 elements
            input3 = torch.tensor(state[2], dtype=torch.float32)
            if input3.ndim == 1:
                input3 = input3.unsqueeze(0)
            else:
                input3 = input3.view(-1, 4)
            
            # Forward pass through the model to get Q-values
            with torch.no_grad():
                q_values = self.model(input1, input2, input3)
            
            # Return the action with the highest Q-value
            return torch.argmax(q_values).item()


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # Unpack batch
        states = [m[0] for m in minibatch]
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = [m[3] for m in minibatch]
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)
        
        # Helper function to process state images.
        def process_state(x):
            # Convert the input to a float tensor.
            t = torch.tensor(x, dtype=torch.float32)
            # If x is 3D (e.g., H, W, C), permute it to (C, H, W)
            if t.dim() == 3:
                return t.permute(2, 0, 1)
            # If x is 4D, assume it has an extra batch dimension (e.g., (1, H, W, C))
            # Remove the extra dimension and then permute.
            elif t.dim() == 4:
                return t.squeeze(0).permute(2, 0, 1)
            else:
                raise ValueError("Unexpected tensor dimension: " + str(t.dim()))
        
        # Process the batches for state[0] and state[1] using the helper.
        input1_batch = torch.stack([process_state(s[0]) for s in states])
        input2_batch = torch.stack([process_state(s[1]) for s in states])
        
        # For state[2] (assumed to be a feature vector), reshape to 2D.
        input3_batch = torch.stack([
            torch.tensor(s[2], dtype=torch.float32).view(-1)
            for s in states
        ])
        
        # Similarly process the next_states.
        next_input1_batch = torch.stack([process_state(ns[0]) for ns in next_states])
        next_input2_batch = torch.stack([process_state(ns[1]) for ns in next_states])
        next_input3_batch = torch.stack([
            torch.tensor(ns[2], dtype=torch.float32).view(-1)
            for ns in next_states
        ])
        
        # Compute current Q values and select Q for chosen actions.
        current_q = self.model(input1_batch, input2_batch, input3_batch)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        current_q_selected = current_q.gather(1, actions_tensor).squeeze(1)
        
        # Compute target Q values.
        with torch.no_grad():
            next_q = self.model(next_input1_batch, next_input2_batch, next_input3_batch)
            max_next_q = next_q.max(1)[0]
            targets = torch.tensor(rewards) + self.gamma * max_next_q * (1 - torch.tensor(dones))
        
        # Compute loss and optimize.
        loss = F.mse_loss(current_q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)
    
    @staticmethod
    def generate_summary(step_time, mean_waiting_time, mean_travel_time,density, average_speed, co2_emission,waiting,queue_length,traffic_light_phase,current_phase, next_phase,reward):
        return {
            "step_time": step_time,
            "mean_waiting_time": mean_waiting_time,
            "mean_travel_time": mean_travel_time,
            "density": density,
            "average_speed": average_speed,
            "co2_emission": co2_emission,
            "waiting_vehicles": waiting,
            "queue_length": queue_length,
            "traffic_light_phase": traffic_light_phase,
            "current_phase": current_phase,
            "next_phase": next_phase,
            "reward": reward
        }

if __name__ == "__main__":
    weights = {
        'average_speed': 0.4,   # Thưởng cho tốc độ cao
        'co2_emission': -0.15,   # Phạt cho CO2 cao
        'waiting_time': -0.4,   # Phạt cho thời gian chờ lâu
        'queue_length': -0.45    # Phạt cho hàng đợi dài
    }

    # Định nghĩa giá trị tối đa để chuẩn hóa
    max_values = {
        'average_speed': 50.0,  # km/h
        'co2_emission': 1000.0, # g/s
        'waiting_time': 1000.0, # giây
        'queue_length': 200.0   # số xe
    }

    nodes = 4
    edges = 4
    lanes = 3

    summaries = []
    trajectory_data = []   
    total_queue_length = []
    total_waiting_time = []
    total_fuel_consumption = []
    
    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml","--quit-on-end"])
    sumoObject.generate_routefile()

    num_episodes = 50
    batch_size = 32

    tlsObject = TLSClass(
        epsilon_start=0.9,      # Start with high exploration
        epsilon_end=0.01,       # End with low exploration
        epsilon_decay_episodes=80  # Decay over 80% of episodes
    )
    
    
    edges = sumoObject.get_edges()
    lanes = sumoObject.get_lanes()


    # Pre-load weights if needed
    '''try:
        tlsObject.load('trained_kernel_initialiser.pth')
    except:
        print("No models found to instantiate kernel")
    '''
    traci.close()

    # Initialize metrics tracking
    episode_rewards = []
    episode_queue_lengths = []
    episode_waiting_times = []


    for episode in range(num_episodes):

        tlsObject.update_epsilon()


        print(f"Starting episode {episode + 1}/{num_episodes}")
        stepz = 0
        reward1 = 0
        reward2 = 0
        queue_length = 0
        waiting_time = 0
        fuel_consumption = 0
        episode_reward = 0

        traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml","--quit-on-end"])
        traci.trafficlight.setPhase("0", 1)
        traci.trafficlight.setPhaseDuration("0", 200)
        complete_program = traci.trafficlight.getCompleteRedYellowGreenDefinition("0")
# Print the phases
        for program_index, program in enumerate(complete_program):
            print(f"Program {program_index}:")
            for phase_index, phase in enumerate(program.phases):
                # Then print the common attributes that should exist
                print(f"Phase {phase_index}: state={phase.state}, duration={phase.duration}")

        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:   # as long as vehicles are there
            traci.simulationStep()
            state = sumoObject.get_state()
            action = tlsObject.act(state)
            light = np.argmax(state[2].reshape(4,))
            step_time = traci.simulation.getCurrentTime() / 1000  # Convert to seconds
            waiting = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
            mean_waiting_time = (
                sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()) 
                / max(len(traci.vehicle.getIDList()), 1)  # Tránh lỗi chia cho 0
            )
            mean_travel_time = (
                sum(traci.vehicle.getAccumulatedWaitingTime(veh) + traci.vehicle.getTimeLoss(veh) for veh in traci.vehicle.getIDList()) 
                / max(len(traci.vehicle.getIDList()), 1)  # Tránh lỗi chia cho 0
            )
            vehicles = traci.vehicle.getIDList()
            total_road_length = sum(traci.lane.getLength(lane) for lane in traci.lane.getIDList())
            density = len(vehicles) / max(total_road_length/1000, 1)  # Tránh chia cho 0
            average_speed1 = sum(traci.vehicle.getSpeed(v) for v in vehicles) / max(len(vehicles), 1)
            average_speed = average_speed1 * 3.6  # Convert m/s to km/h 
            co2_mg_per_sec  = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
            co2_emission =  co2_mg_per_sec / 1000  # Chuyển mg/s → g/s
            queue_length= sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
            traffic_light_phase = traci.trafficlight.getPhase("0")
            current_phase = traci.trafficlight.getPhase("0")
            next_phase = (current_phase + 1) % 4  # Assuming 4 phases

                         # Chuẩn hóa các giá trị
            norm_average_speed = average_speed / max_values['average_speed']
            norm_co2_emission = co2_emission / max_values['co2_emission']
            norm_waiting_time = waiting / max_values['waiting_time'] 
            norm_queue_length = queue_length / max_values['queue_length']

            # Tính phần thưởng mới
            reward = (weights['average_speed'] * norm_average_speed +
                      weights['co2_emission'] * norm_co2_emission +
                      weights['waiting_time'] * norm_waiting_time +
                      weights['queue_length'] * norm_queue_length)
            summary = TLSClass.generate_summary(step_time, mean_waiting_time, mean_travel_time,density, average_speed, co2_emission,waiting,queue_length,traffic_light_phase,current_phase, next_phase,reward)
            summaries.append(summary)

            
            # Action handling for traffic light control
            if(action == 0):  # Choose phase 0 (GGGGgrrrrrGGGGgrrrrr)
                if(light == 0):  # Already in phase 0
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 31)  # Match duration from XML
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light == 1):  # Need to transition from phase 1 to phase 0
                    for i in range(4):  # Use phase 7 (yellow transition)
                        stepz += 1
                        traci.trafficlight.setPhase('0', 7)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light in [2, 3, 4, 5]):  # Need to transition properly from other phases
                    # First finish any ongoing special phase if needed
                    if light in [2, 3]:  # Special turn phases
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 6)  # Phase 6 (yellow transition)
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    elif light in [4, 5]:  # Special turn phases
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 7)  # Phase 7 (yellow transition)
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    
                    # Now go to phase 0
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 0)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

            if(action == 1):  # Choose phase 1 (rrrrrGGGGgrrrrrGGGGg)
                if(light == 1):  # Already in phase 1
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light == 0):  # Need to transition from phase 0 to phase 1
                    for i in range(4):  # Use phase 6 (yellow transition)
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                        
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light in [2, 3, 4, 5]):  # Need to transition properly from other phases
                    # First finish any ongoing special phase if needed
                    if light in [2, 3]:  # Special turn phases 
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 6)  # Yellow transition
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    elif light in [4, 5]:  # Special turn phases
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 7)  # Yellow transition
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    
                    # Now go to phase 1
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 31)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                    if(action == 2):  # Choose phase 2 (rrGGrGrrrrrrGGrGrrrr)
                        if(light == 2):  # Already in phase 2
                            for i in range(15):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 2)
                                traci.trafficlight.setPhaseDuration("0", 15)
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light == 0 or light == 1):  # Transition from main phases to special turn phase
                    # First go to yellow transition
                    for i in range(4):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 6)  # Yellow transition
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                    
                    # Then go to phase 2
                    for i in range(15):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 2)
                        traci.trafficlight.setPhaseDuration("0", 15)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                
                elif(light in [3, 4, 5]):  # Transition from other special phases
                    # First finish any ongoing special phase with yellow if needed
                    if light in [3]:
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 6)  # Yellow transition
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    elif light in [4, 5]:
                        for i in range(2):
                            stepz += 1
                            traci.trafficlight.setPhase('0', 7)  # Yellow transition
                            queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                        traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                            waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                        traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                            fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                            traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                            traci.simulationStep()
                    
                    # Then go to phase 2
                    for i in range(15):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 2)
                        traci.trafficlight.setPhaseDuration("0", 15)
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()

                    if(action == 3):  # Choose phase 3 (GrrrrrrGGrGrrrrrrGGr)
                        if(light == 3):  # Already in phase 3
                            for i in range(15):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 3)
                                traci.trafficlight.setPhaseDuration("0", 15)
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                traci.simulationStep()
                        
                        elif(light == 0 or light == 1):  # Transition from main phases to special turn phase
                            # First go to yellow transition
                            for i in range(4):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 7)  # Yellow transition
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                traci.simulationStep()
                            
                            # Then go to phase 3
                            for i in range(15):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 3)
                                traci.trafficlight.setPhaseDuration("0", 15)
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                traci.simulationStep()
                        
                        elif(light in [2, 4, 5]):  # Transition from other special phases
                            # First finish any ongoing special phase with yellow if needed
                            if light in [2]:
                                for i in range(2):
                                    stepz += 1
                                    traci.trafficlight.setPhase('0', 7)  # Yellow transition
                                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                    traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                    traci.simulationStep()
                            elif light in [4, 5]:
                                for i in range(2):
                                    stepz += 1
                                    traci.trafficlight.setPhase('0', 6)  # Yellow transition
                                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                    traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                    traci.simulationStep()
                            
                            # Then go to phase 3
                            for i in range(15):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 3)
                                traci.trafficlight.setPhaseDuration("0", 15)
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                traci.simulationStep()

                    if(action == 4):  # Choose phase 4 (GrrrGGrrrrGrrrGGrrrr)
                        if(light == 4):  # Already in phase 4
                            for i in range(10):
                                stepz += 1
                                traci.trafficlight.setPhase('0', 4)
                                traci.trafficlight.setPhaseDuration("0", 10)
                                queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                            traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                            traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                traci.simulationStep()
                        
                        else:  # Need to transition from any other phase
                            # First go to phase 0 or 1 if needed
                            if light not in [0, 1]:
                                # Transition to phase 0 with appropriate yellow
                                if light in [2, 3]:
                                    for i in range(4):
                                        stepz += 1
                                        traci.trafficlight.setPhase('0', 7)  # yellow transition
                                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                        traci.simulationStep()
                                else:  # light in [4, 5]
                                    for i in range(4):
                                        stepz += 1
                                        traci.trafficlight.setPhase('0', 6)  # yellow transition
                                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                    traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                    traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                        traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                        traci.simulationStep()

                                # Then go to phase 0
                                for i in range(5):
                                    stepz += 1
                                    traci.trafficlight.setPhase('0', 0)
                                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                    traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                    traci.simulationStep()

                                # Finally transition to phase 4
                                for i in range(4):
                                    stepz += 1
                                    traci.trafficlight.setPhase('0', 7)  # yellow transition from phase 0 to phase 4
                                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                    traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                    traci.simulationStep()

                                for i in range(10):
                                    stepz += 1
                                    traci.trafficlight.setPhase('0', 4)
                                    traci.trafficlight.setPhaseDuration("0", 10)
                                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + 
                                                traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + 
                                                traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + 
                                                    traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                                    traci.simulationStep()

            new_state = sumoObject.get_state()
            tlsObject.remember(state, action, reward, new_state, False)
            if len(tlsObject.memory) > batch_size:
                tlsObject.replay(batch_size)

        mem = tlsObject.memory[-1]
        del tlsObject.memory[-1]
        tlsObject.memory.append((mem[0], mem[1], reward, mem[3], True))
        
        total_queue_length.append(f"{episode} : {queue_length}, {waiting_time}, {fuel_consumption}")
        traci.close(wait=False)
    with open('simulation_summary.csv', 'w', newline='') as csvfile:
        fieldnames = ["step_time", "mean_waiting_time", "mean_travel_time","density","average_speed","co2_emission","waiting_vehicles","queue_length","traffic_light_phase","current_phase","next_phase","reward"]    
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "step_time": round(summary["step_time"], 2),
                "mean_waiting_time": round(summary["mean_waiting_time"], 2),
                "mean_travel_time": round(summary["mean_travel_time"], 2),
                "density": round(summary["density"], 2),
                "average_speed": round(summary["average_speed"], 2),
                "co2_emission": round(summary["co2_emission"], 2),
                "waiting_vehicles": round(summary["waiting_vehicles"], 2),
                "queue_length": round(summary["queue_length"], 2),
                "traffic_light_phase": summary["traffic_light_phase"],
                "current_phase": summary["current_phase"],
                "next_phase": summary["next_phase"],
                "reward": round(summary["reward"], 2)
            })
    # with open('trajectory_data.csv', 'w', newline='') as csvfile:
    #     fieldnames = ["time", "density", "average_speed","co2_emission", "waiting_vehicles","queue_length", "traffic_light_phase", "current_phase", "next_phase"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     for data in trajectory_data:
    #         writer.writerow({
    #             "time": round(data["state"]["time"], 2),
    #             "density": round(data["state"]["density"], 2),
    #             "average_speed": round(data["state"]["average_speed"], 2),
    #             "co2_emission": round(data["state"]["co2_emission"], 2),
    #             "waiting_vehicles": round(data["state"]["waiting_vehicles"], 2),
    #             "queue_length": round(data["state"]["queue_length"], 2),
    #             "traffic_light_phase": data["state"]["traffic_light_phase"],
    #             "current_phase": data["action"]["current_phase"],
    #             "next_phase": data["action"]["next_phase"]
    #         })
    for queue_length in total_queue_length:
        print(queue_length)

    #Save final model
    tlsObject.save('trained_kernel_initialiser_2.pth')

    sys.stdout.flush()