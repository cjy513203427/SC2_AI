# -*- encoding: utf-8 -*-
'''
@File    :   demo1.py
@Modify Time      @Author       @Desciption
------------      -------       -----------
2019/11/3 12:32   Jonas           None
'''
import math
import os
import time

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import *
import random
import numpy as np
import cv2
import keras

HEADLESS = False


# os.environ["SC2PATH"] = 'F:\StarCraft II'

class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False):
        # 经过计算，每分钟大约165迭代次数
        # self.ITERATIONS_PER_MINUTE = 165
        # 最大农民数量
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        self.scouts_and_spots = {}

        if self.use_model:
            print("use model")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")

    # 存储.npy训练数据
    # def on_end(self, game_result):
    #     print('--- on_end called ---')
    #     print(game_result)
    #
    #     if game_result == Result.Victory:
    #         np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("log.txt", "a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

    async def on_step(self, iteration):
        # self.iteration = iteration
        self.timeMinutes = ((self.state.game_loop / 22.4) / 60)

        # print('Time:', self.timeMinutes)
        await self.build_scout()
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()

    async def build_scout(self):
        if len(self.units(UnitTypeId.OBSERVER)) < math.floor(self.timeMinutes / 3):
            for rf in self.units(UnitTypeId.ROBOTICSFACILITY).ready.noqueue:
                print(len(self.units(UnitTypeId.OBSERVER)), self.timeMinutes / 3)
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(UnitTypeId.OBSERVER))

    ## 侦察
    async def scout(self):
        # {DISTANCE_TO_ENEMY_START:EXPANSIONLOC}
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(UnitTypeId.ROBOTICSFACILITY).ready) == 0:
            unit_type = UnitTypeId.PROBE
            unit_limit = 1
        else:
            unit_type = UnitTypeId.OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == UnitTypeId.PROBE:
            for unit in self.units(UnitTypeId.PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        # 分配侦察任务
        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == UnitTypeId.PROBE:
                                        for unit in self.units(UnitTypeId.PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        # 防止去侦察的农民采矿
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(UnitTypeId.PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UnitTypeId作为key,半径和颜色是value
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.ROBOTICSFACILITY: [5, (215, 155, 0)],

            UnitTypeId.VOIDRAY: [3, (255, 100, 0)],
            # OBSERVER: [3, (255, 255, 255)],
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        # 主基地名称
        main_base_names = ['nexus', 'commandcenter', 'orbitalcommand', 'planetaryfortress', 'hatchery']
        # 记录敌方基地位置
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            # 不是主基地建筑，画小一些
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe", "scv", "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(UnitTypeId.OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # 追踪资源、人口和军队人口比
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        # 农民/人口      worker/supply ratio
        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)
        # 人口/200    plausible supply (supply/200.0)
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200), 3)
        # (人口-现有人口)/人口  population ratio (supply_left/supply)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150), 3)
        # 气体/1500   gas/1500
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)
        # 晶体矿/1500  minerals minerals/1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not HEADLESS:
            if self.use_model:
                cv2.imshow('Model Intel', resized)
                cv2.waitKey(1)
            else:
                cv2.imshow('Random Intel', resized)
                cv2.waitKey(1)

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    # 建造农民
    async def build_workers(self):
        # 星灵枢钮*16（一个基地配备16个农民）大于农民数量并且现有农民数量小于MAX_WORKERS
        if len(self.units(UnitTypeId.NEXUS)) * 16 > len(self.units(UnitTypeId.PROBE)) and len(
                self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            # 星灵枢纽(NEXUS)无队列建造，可以提高晶体矿的利用率，不至于占用资源
            for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
                # 是否有50晶体矿建造农民
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))

    ## 建造水晶
    async def build_pylons(self):
        ## 供应人口和现有人口之差小于5且建筑不是正在建造
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)

    ## 建造吸收厂
    async def build_assimilators(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            # 在瓦斯泉上建造吸收厂
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))

    ## 开矿
    async def expand(self):
        # (self.iteration / self.ITERATIONS_PER_MINUTE)是一个缓慢递增的值,动态开矿
        if self.units(UnitTypeId.NEXUS).amount < self.timeMinutes / 2 and self.can_afford(
                UnitTypeId.NEXUS):
            await self.expand_now()

    ## 建造进攻性建筑
    async def offensive_force_buildings(self):
        # print(self.iteration / self.ITERATIONS_PER_MINUTE)
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random
            # 根据神族建筑科技图，折跃门建造过后才可以建造控制核心
            if self.units(UnitTypeId.GATEWAY).ready.exists and not self.units(UnitTypeId.CYBERNETICSCORE):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
            # 否则建造折跃门
            # (self.iteration / self.ITERATIONS_PER_MINUTE)/2 是一个缓慢递增的值
            # elif len(self.units(UnitTypeId.GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
            elif len(self.units(UnitTypeId.GATEWAY)) < 1:
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)
            # 控制核心存在的情况下建造机械台
            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.ROBOTICSFACILITY)) < 1:
                    if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(
                            UnitTypeId.ROBOTICSFACILITY):
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)

            # 控制核心存在的情况下建造星门
            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.STARGATE)) < self.timeMinutes:
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon)

    ## 造兵
    async def build_offensive_force(self):
        # 无队列化建造
        # for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue:
        #     if not self.units(UnitTypeId.STALKER).amount > self.units(UnitTypeId.VOIDRAY).amount:
        #
        #         if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
        #             await self.do(gw.train(UnitTypeId.STALKER))

        for sg in self.units(UnitTypeId.STARGATE).ready.noqueue:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(UnitTypeId.VOIDRAY))

    ## 寻找目标
    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            # 随机选取敌方单位
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_units) > 0:
            # 随机选取敌方建筑
            return random.choice(self.known_enemy_structures)
        else:
            # 返回敌方出生点位
            return self.enemy_start_locations[0]

    ## 进攻
    async def attack(self):
        # # {UNIT: [n to fight, n to defend]}
        # aggressive_units = {UnitTypeId.VOIDRAY: [8, 3]}
        #
        # for UNIT in aggressive_units:
        #     # 攻击模式
        #     if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][
        #         1]:
        #         for s in self.units(UNIT).idle:
        #             await self.do(s.attack(self.find_target(self.state)))
        #     # 防卫模式
        #     elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
        #         if len(self.known_enemy_units) > 0:
        #             for s in self.units(UNIT).idle:
        #                 await self.do(s.attack(random.choice(self.known_enemy_units)))
        if len(self.units(UnitTypeId.VOIDRAY).idle) > 0:

            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape(-1, 176, 200, 3)])
                choice = np.argmax(prediction[0])

                choice_dict = {0: "No Attack!",
                               1: "Attack close to our nexus!",
                               2: "Attack Enemy Structure!",
                               3: "Attack Enemy Start!"}
                print("Choice #{}:{}".format(choice, choice_dict[choice]))
            else:
                choice = random.randrange(0, 4)

            target = False
            if self.timeMinutes > self.do_something_after:
                if choice == 0:
                    # 什么都不做
                    wait = random.randrange(7, 100) / 100
                    self.do_something_after = self.timeMinutes + wait

                elif choice == 1:
                    # 攻击离星灵枢纽最近的单位
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))

                elif choice == 2:
                    # 攻击敌方建筑
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # 攻击敌方出生位置（换家）
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(UnitTypeId.VOIDRAY).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y, self.flipped])


## 启动游戏
# for i in range(50):
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot(use_model=True)),
    Computer(Race.Terran, Difficulty.Medium)
], realtime=False)
