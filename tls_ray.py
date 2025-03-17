from __future__ import absolute_import, print_function

import csv
import os
import random
import sys
from collections import deque

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from sumolib import checkBinary

from sumoClass import SumoClass


class TrafficModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x1 = self.conv1(x[:, 0:1, :, :])
        x1 = torch.relu(x1)
        x1 = self.conv2(x1)
        x1 = torch.relu(x1)
        x1 = self.flatten(x1)

        x2 = self.conv1(x[:, 1:2, :, :])
        x2 = torch.relu(x2)
        x2 = self.conv2(x2)
        x2 = torch.relu(x2)
        x2 = self.flatten(x2)

        x3 = self.flatten(x[:, 2:3, :, :])

        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x, state

ModelCatalog.register_custom_model("traffic_model", TrafficModel)

class TrafficEnv:
    def __init__(self, config):
        self.sumoObject = SumoClass(config["nodes"], config["edges"], config["lanes"])
        self.sumoBinary = checkBinary('sumo-gui' if not config["nogui"] else 'sumo')
        self.num_episodes = config["num_episodes"]
        self.batch_size = config["batch_size"]
        self.action_size = 4
        self.gamma = 0.95
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.memory = deque(maxlen=4000)

    def reset(self):
        traci.start([self.sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        self.sumoObject.generate_routefile()
        self.state = self.sumoObject.get_state()
        return self.state
    def generate_summary(self):
        simulation_time = traci.simulation.getTime()
        loaded_vehicle_number = traci.simulation.getLoadedNumber()
        inserted_vehicle_number = traci.simulation.getDepartedNumber()
        running_vehicle_number = traci.simulation.getRunningNumber()
        waiting_vehicle_number = traci.simulation.getWaitingTime()
        ended_vehicle_number = traci.simulation.getArrivedNumber()
        mean_waiting_time = traci.simulation.getMeanWaitingTime()
        mean_travel_time = traci.simulation.getMeanTravelTime()

        summary = [
            simulation_time,
            loaded_vehicle_number,
            inserted_vehicle_number,
            running_vehicle_number,
            waiting_vehicle_number,
            ended_vehicle_number,
            mean_waiting_time,
            mean_travel_time
        ]

        with open('summary.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summary)

    def step(self, action):
        reward1 = 0
        reward2 = 0
        queue_length = 0
        waiting_time = 0
        fuel_consumption = 0
        stepz = 0

        light = np.argmax(self.state[2].reshape(4,))
        if(action == 0):
            if(light == 0):
                reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
    
            if(light == 1):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 6)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()

                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                
            if(light == 2):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 7)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                
                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
            
            if(light == 3):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 7)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                
                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                        
        if(action == 1):
            if(light == 1):
                reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()

            if(light == 0):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 7)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()

                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()

            if(light == 2):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 6)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
            
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                        self.generate_summary()
            
            if(light == 3):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 6)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                    
                    # Action Execution
                    reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    for i in range(30):
                        stepz += 1
                        traci.trafficlight.setPhase('0', 1)
                        traci.trafficlight.setPhaseDuration("0", 50)
                        reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                        reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                        queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                        waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                        fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                        traci.simulationStep()
                        self.generate_summary()
                        
        if(action == 2):
            if(light == 0 or light == 1):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 6)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                
                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                reward2 = traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('3i') + traci.edge.getLastStepVehicleNumber('4i')
                    reward2 += traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()

        if(action == 3):
            if(light == 0 or light == 1):
                for i in range(2):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 7)
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()
                
                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward2 = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                for i in range(30):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 1)
                    traci.trafficlight.setPhaseDuration("0", 50)
                    reward1 += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')
                    queue_length += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    waiting_time += (traci.edge.getWaitingTime('1i') + traci.edge.getWaitingTime('2i') + traci.edge.getWaitingTime('3i') + traci.edge.getWaitingTime('4i'))
                    fuel_consumption += (traci.edge.getFuelConsumption('1i') + traci.edge.getFuelConsumption('2i') + traci.edge.getFuelConsumption('3i') + traci.edge.getFuelConsumption('4i'))
                    traci.simulationStep()
                    self.generate_summary()


        new_state = self.sumoObject.get_state()
        reward = reward1 - reward2
        done = stepz >= 5000 or traci.simulation.getMinExpectedNumber() <= 0
        return new_state, reward, done, {}

    def close(self):
        traci.close(wait=False)

if __name__ == "__main__":
    ray.init()

    config = {
        "env": TrafficEnv,
        "env_config": {
            "nodes": 4,
            "edges": 4,
            "lanes": 3,
            "nogui": False,
            "num_episodes": 50,
            "batch_size": 32
        },
        "model": {
            "custom_model": "traffic_model"
        },
        "num_workers": 1,
        "framework": "torch",
        "experimental": "_validate_config=False"
    }

    tune.run(PPO, config=config)