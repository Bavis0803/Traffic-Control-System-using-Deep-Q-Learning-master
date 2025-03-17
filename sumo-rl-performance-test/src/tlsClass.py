from sumoClass import SumoClass
from tlsClass import TLSClass
from sumolib import checkBinary
import traci
import numpy as np

def test_performance(trained_model_path, num_episodes=10):
    nodes = 4
    edges = 4
    lanes = 3

    sumoObject = SumoClass(nodes, edges, lanes)
    options = sumoObject.get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "data/cross.sumocfg", "--start", "--tripinfo-output", "tripinfo.xml", "--quit-on-end"])
    sumoObject.generate_routefile()

    tlsObject = TLSClass()
    tlsObject.load(trained_model_path)

    performance_metrics = []

    for episode in range(num_episodes):
        print(f"Testing episode {episode + 1}/{num_episodes}")
        stepz = 0
        total_waiting_time = 0
        total_queue_length = 0

        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 5000:
            traci.simulationStep()
            state = sumoObject.get_state()
            action = tlsObject.act(state)

            waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
            queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())

            total_waiting_time += waiting_time
            total_queue_length += queue_length
            stepz += 1

        average_waiting_time = total_waiting_time / max(stepz, 1)
        average_queue_length = total_queue_length / max(stepz, 1)

        performance_metrics.append({
            "episode": episode + 1,
            "average_waiting_time": average_waiting_time,
            "average_queue_length": average_queue_length
        })

    traci.close()
    return performance_metrics

if __name__ == "__main__":
    trained_model_path = 'trained_kernel_initialiser.pth'  # Path to the trained model
    results = test_performance(trained_model_path)
    for result in results:
        print(f"Episode {result['episode']}: Average Waiting Time: {result['average_waiting_time']}, Average Queue Length: {result['average_queue_length']}")