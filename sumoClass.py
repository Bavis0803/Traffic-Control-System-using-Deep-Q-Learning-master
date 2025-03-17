from __future__ import absolute_import, print_function

import optparse
import os
import random
import sys

import numpy as np
import torch
import traci


class SumoClass:
    
    def __init__(self, nodes, edges, lanes):
        self.nodes = nodes
        self.edges = edges
        self.lanes = lanes

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options
    
    def get_edges(self):
        edges = traci.edge.getIDList()
        edges = np.asarray(edges)
        return edges[0:7:2]

    def get_lanes(self):
        lanes = []
        edges = self.get_edges()
        for edge in edges:
            for i in range(self.lanes):
                lanes.append(edge + '_' + str(i))
        return lanes
    
    def generate_routefile(self):
        N = 3600  # Giữ N = 1800 (30 phút mô phỏng)

        pWE = 1. / 10
        pEW = 1. / 10
        pNS = 1. / 30
        pSN = 1. / 30
        
        pWN = 1. / 20
        pWS = 1. / 20
        pEN = 1. / 20
        pES = 1. / 20
        
        pNW = 1. / 40
        pSW = 1. / 40
        pNE = 1. / 40
        pSE = 1. / 40

        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
                <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
                <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="20" guiShape="bus"/>
                
                <route id="right" edges="51o 1i 2o 52i" />
                <route id="left" edges="52o 2i 1o 51i" />
                <route id="down" edges="54o 4i 3o 53i" />
                <route id="up" edges="53o 3i 4o 54i" />
                
                <route id="right_up" edges="51o 1i 4o 54i" />
                <route id="right_down" edges="51o 1i 3o 53i" />
                <route id="left_up" edges="52o 2i 4o 54i" />
                <route id="left_down" edges="52o 2i 3o 53i" />
                
                <route id="up_right" edges="53o 3i 2o 52i" />
                <route id="up_left" edges="53o 3i 1o 51i" />
                <route id="down_right" edges="54o 4i 2o 52i" />
                <route id="down_left" edges="54o 4i 1o 51i" />
                """, file=routes)
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pWN:
                    print('    <vehicle id="left_up_%i" type="typeWE" route="left_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pWS:
                    print('    <vehicle id="left_down_%i" type="typeWE" route="left_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEN:
                    print('    <vehicle id="right_up_%i" type="typeWE" route="right_up" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pES:
                    print('    <vehicle id="right_down_%i" type="typeWE" route="right_down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                
                if random.uniform(0, 1) < pSE:
                    print('    <vehicle id="down_right_%i" type="typeNS" route="down_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pSW:
                    print('    <vehicle id="down_left_%i" type="typeNS" route="down_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNE:
                    print('    <vehicle id="up_right_%i" type="typeNS" route="up_right" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNW:
                    print('    <vehicle id="up_left_%i" type="typeNS" route="up_left" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1

            print("</routes>", file=routes)
            print(f"Tổng số xe được tạo trong episode: {vehNr}")  # In số lượng xe để kiểm tra

    def get_state(self):
        edges = self.get_edges()
        lanes = self.get_lanes()
        
        cellLength = 7
        offset = 11
        speedLimit = 20
        
        positionMatrix = []
        velocityMatrix = []
        
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)
        
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')
        
        junctionPosition = traci.junction.getPosition('0')[0]  # (510, 510)
        
        for v in vehicles_road1:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if ind < 12:
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if ind < 12:
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[1]

        for v in vehicles_road3:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if ind < 12:
                positionMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[8 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        
        for v in vehicles_road4:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if ind < 12:
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        # Sửa phần xử lý lights
        light = []
        phase = traci.trafficlight.getPhase('0')
        num_phases = len(traci.trafficlight.getAllProgramLogics("0")[0].phases)
        if 0 <= phase < num_phases:
            # Ánh xạ pha đèn thành vector [1, 0, 0, 0] dựa trên chỉ số pha
            light = [0] * num_phases
            light[phase] = 1
        else:
            print(f"Cảnh báo: Pha đèn không hợp lệ: {phase}. Gán giá trị mặc định.")
            light = [0] * num_phases  # Mặc định tất cả đều 0
        
        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)
        
        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)
        
        lights = np.array(light)
        if lights.size == 0:
            raise ValueError("The lights array is empty and cannot be reshaped.")
        lights = lights.reshape(1, num_phases, 1)

        return [position, velocity, lights]
    def get_state_irl(self):
        """ Trả về trạng thái giao thông hiện tại cho mô hình IRL """
        try:
            step_time = traci.simulation.getCurrentTime() / 1000  # Convert to seconds

            vehicles = traci.vehicle.getIDList()
            
            # Tính tốc độ trung bình (km/h)
            average_speed = sum(traci.vehicle.getSpeed(v) for v in vehicles) / max(len(vehicles), 1) * 3.6
            
            # Tính tổng lượng khí thải CO2 (kg/s)
            co2_emission = sum(traci.vehicle.getCO2Emission(v) for v in vehicles) / 1000
            
            # Đếm số xe đang chờ (tốc độ < 0.1 m/s)
            waiting_vehicles = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
            
            # Tính độ dài hàng đợi (tổng số xe dừng trên tất cả các cạnh)
            queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
            
            # Lấy pha đèn hiện tại
            current_phase = traci.trafficlight.getPhase("0")
            
            # Tính pha tiếp theo (giả sử có 4 pha: 0, 1, 2, 3)
            num_phases = len(traci.trafficlight.getAllProgramLogics("0")[0].phases)
            next_phase = (current_phase + 1) % num_phases
            
            return {
                "step_time": step_time,
                "average_speed": average_speed,
                "co2_emission": co2_emission,
                "waiting_vehicles": waiting_vehicles,
                "queue_length": queue_length,
                "t_current_phase": current_phase,
                "t_next_phase": next_phase
            }
        except traci.TraCIException as e:
            print(f"Lỗi khi truy xuất dữ liệu từ SUMO: {e}")
            return {
                "step_time": 0,
                "average_speed": 0,
                "co2_emission": 0,
                "waiting_vehicles": 0,
                "queue_length": 0,
                "t_current_phase": 0,
                "t_next_phase": 0
            }

    def get_possible_actions(self):
        """ Lấy danh sách các phase đèn giao thông có thể thực hiện """
        traffic_light_id = "0"
        
        # 📌 Lấy số phase đúng cách
        num_phases = len(traci.trafficlight.getAllProgramLogics(traffic_light_id)[0].phases)
        
        return list(range(num_phases))  # Trả về danh sách phase: [0, 1, 2, ..., num_phases-1]


    def apply_action(self, action):
        """ Thay đổi pha đèn tín hiệu với logic hợp lý """
        traffic_light_id = "0"

        # Lấy thông tin logic của đèn giao thông hiện tại
        logic = traci.trafficlight.getAllProgramLogics(traffic_light_id)[0]

        # Cập nhật logic để đổi pha đúng cách
        for phase in logic.phases:
            if phase.state == action:  # Tìm pha phù hợp với action
                traci.trafficlight.setPhase(traffic_light_id, action)
                return

    #     print(f"⚠️ Action {action} không hợp lệ cho đèn {traffic_light_id}")
    # def apply_action(self, action, traffic_light_id="0"):
    #     """ Thay đổi pha đèn tín hiệu với logic hợp lý """
    #     try:
    #         logic = traci.trafficlight.getAllProgramLogics(traffic_light_id)[0]
    #         num_phases = len(logic.phases)
    #         if 0 <= action < num_phases:
    #             traci.trafficlight.setPhase(traffic_light_id, action)
    #         else:
    #             print(f"⚠️ Action {action} không hợp lệ cho đèn {traffic_light_id}. Số pha khả dụng: {num_phases}")
    #             current_phase = traci.trafficlight.getPhase(traffic_light_id)
    #             traci.trafficlight.setPhase(traffic_light_id, current_phase)
    #     except traci.TraCIException as e:
    #         print(f"⚠️ Lỗi khi áp dụng hành động: {e}")
    # def apply_action(self, action):
    #     """Thay đổi pha đèn tín hiệu với logic hợp lý"""
    #     traffic_light_id = "0"
    #     try:
    #         Lấy thông tin logic của đèn giao thông
    #         logic = traci.trafficlight.getAllProgramLogics(traffic_light_id)[0]
            
    #         Kiểm tra xem action có hợp lệ không
    #         num_phases = len(logic.phases)
    #         if 0 <= action < num_phases:
    #             traci.trafficlight.setPhase(traffic_light_id, action)
    #         else:
    #             print(f"⚠️ Action {action} không hợp lệ cho đèn {traffic_light_id}. Số pha khả dụng: {num_phases}")
    #             Giữ pha hiện tại nếu action không hợp lệ
    #             current_phase = traci.trafficlight.getPhase(traffic_light_id)
    #             traci.trafficlight.setPhase(traffic_light_id, current_phase)
    #     except traci.TraCIException as e:
    #         print(f"⚠️ Lỗi khi áp dụng hành động: {e}")


