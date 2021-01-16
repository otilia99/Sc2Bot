import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import UnitTypeId, AbilityId, UpgradeId, BuffId
import random
import numpy as np
import cv2
import os
# import keras
# import tensorflow

os.environ["SC2PATH"] = 'D:/Downloads/Starcraft II/Starcraft II'


class ProtossBot(sc2.BotAI):
    def __init__(self):
        super().__init__()
        # number of iterations per minute
        self.IT_PER_MIN = 165
        self.MAX_WORKERS = 80
        self.do_smth_after_this = 0
        self.train_data = []
        self.scouting_done = False

    # def save_end_result(self, game_result):
    #     if game_result == Result.Victory:
    #         np.save('/results.txt', self.train_data)

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.scout()
        await self.draw_base()
        await self.attack()

        if iteration == 0:
            await self.chat_send("hi (pylon)(glhf)")

    async def draw_base(self):
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [3, (255, 100, 0)],
            UnitTypeId.ROBOTICSFACILITY: [5, (215, 155, 0)]
        }

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        possible_bases = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in possible_bases:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(UnitTypeId.OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        cv2.imshow('RN stuff', resized)
        cv2.waitKey(1)

    async def build_workers(self):
        if (len(self.units(UnitTypeId.NEXUS)) * 16) > len(self.units(UnitTypeId.PROBE)) and \
                len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(UnitTypeId.NEXUS).ready.idle:
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first, placement_step=5)

    async def build_assimilators(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vespene))

    async def expand(self):
        if self.units(UnitTypeId.NEXUS).amount < 5 and \
                self.can_afford(UnitTypeId.NEXUS):
            await self.expand_now()

    async def scout(self):
        if not self.scouting_done:
            if len(self.units(UnitTypeId.OBSERVER)) > 0:
                scout = self.units(UnitTypeId.OBSERVER)[0]
                if scout.is_idle:
                    enemy_location = self.enemy_start_locations[0]
                    go_scout_at = self.random_location_variance(enemy_location)
                    self.scouting_done = True
                    await self.do(scout.move(go_scout_at))
            else:
                for rf in self.units(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                    if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                        await self.do(rf.train(UnitTypeId.OBSERVER))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        send_to = position.Point2(position.Pointlike((x, y)))
        return send_to

    async def offensive_force_buildings(self):
        # for nexus in self.townhalls.ready:
        #     if nexus.energy >= 50:
        #         nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)

        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random

            if self.units(UnitTypeId.GATEWAY).ready.exists and not self.units(UnitTypeId.CYBERNETICSCORE):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon, placement_step=1)

            elif len(self.units(UnitTypeId.GATEWAY)) < 3:
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon, placement_step=2)

            # create stargates
            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if len(self.units(UnitTypeId.STARGATE)) < 4:
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon, placement_step=2)

                if len(self.units(UnitTypeId.ROBOTICSFACILITY)) < 1:
                    if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and \
                            not self.already_pending(UnitTypeId.ROBOTICSFACILITY):
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon, placement_step=2)

            #     if self.can_afford(AbilityId.RESEARCH_WARPGATE) \
            #             and not self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH):
            #         cybercore = self.units(UnitTypeId.CYBERNETICSCORE).ready.first
            #         await self.do(cybercore.research(UpgradeId.WARPGATERESEARCH))
            #
            #     if len(self.units(UnitTypeId.TWILIGHTCOUNCIL)) < 1:
            #         if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and \
            #                 not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL):
            #             await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon, placement_step=2)
            #
            # if self.units(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
            #     if self.can_afford(AbilityId.RESEARCH_BLINK) \
            #             and not self.already_pending_upgrade(UpgradeId.BLINKTECH):
            #         tcouncil = self.units(UnitTypeId.CYBERNETICSCORE).ready.first
            #         await self.do(tcouncil.research(UpgradeId.BLINKTECH))

    async def build_offensive_force(self):
        for gw in self.units(UnitTypeId.GATEWAY).ready.idle:
            if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
                await self.do(gw.train(UnitTypeId.STALKER))
            if self.can_afford(UnitTypeId.SENTRY) and self.supply_left > 0 \
                    and self.units(UnitTypeId.STALKER).amount > 5:
                await self.do(gw.train(UnitTypeId.SENTRY))

        for sg in self.units(UnitTypeId.STARGATE).ready.idle:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(UnitTypeId.VOIDRAY))
            if self.can_afford(UnitTypeId.PHOENIX) and self.supply_left > 0 and \
                    self.units(UnitTypeId.VOIDRAY).amount > 6:
                await self.do(sg.train(UnitTypeId.PHOENIX))
            if self.can_afford(UnitTypeId.ORACLE) and self.supply_left > 0 \
                    and self.units(UnitTypeId.ORACLE).amount < 1:
                await self.do(sg.train(UnitTypeId.ORACLE))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    # async def attack(self):
    #     if self.units(UnitTypeId.STALKER).amount > 15 or self.units(UnitTypeId.VOIDRAY).amount > 10:
    #         for s in self.units(UnitTypeId.STALKER).idle:
    #             await self.do(s.attack(self.find_target(self.state)))
    #         for vd in self.units(UnitTypeId.VOIDRAY).idle:
    #             await self.do(vd.attack(self.find_target(self.state)))
    #         for ph in self.units(UnitTypeId.PHOENIX).idle:
    #             await self.do(ph.attack(self.find_target(self.state)))
    #         for sen in self.units(UnitTypeId.SENTRY).idle:
    #             await self.do(sen.attack(self.find_target(self.state)))
    #
    #     elif self.units(UnitTypeId.STALKER).amount > 6 or self.units(UnitTypeId.VOIDRAY).amount > 6:
    #         if len(self.known_enemy_units) > 0:
    #             for s in self.units(UnitTypeId.STALKER).idle:
    #                 await self.do(s.attack(random.choice(self.known_enemy_units)))
    #             for vd in self.units(UnitTypeId.VOIDRAY).idle:
    #                 await self.do(vd.attack(random.choice(self.known_enemy_units)))
    #             for ph in self.units(UnitTypeId.PHOENIX).idle:
    #                 await self.do(ph.attack(random.choice(self.known_enemy_units)))
    #             for sen in self.units(UnitTypeId.SENTRY).idle:
    #                 await self.do(sen.attack(random.choice(self.known_enemy_units)))
    async def attack(self):
        attack_units = {
                UnitTypeId.STALKER: [15, 5],
                UnitTypeId.VOIDRAY: [8, 3],
                UnitTypeId.PHOENIX: [8, 2],
                UnitTypeId.SENTRY: [4, 3],
                UnitTypeId.ZEALOT: [10, 5]}

        for u in attack_units:
            if len(self.units(u).idle) > 0:
                choice = random.randrange(0, 4)
                target = False
                if self.iteration > self.do_smth_after_this:
                    if choice == 0:
                        wait = random.randrange(20, 165)
                        self.do_smth_after_this = self.iteration + wait

                    elif choice == 1:
                        if len(self.known_enemy_units) > 0:
                            target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))

                    elif choice == 2:
                        if len(self.known_enemy_structures) > 0:
                            target = random.choice(self.known_enemy_structures)

                    elif choice == 3:
                        target = self.enemy_start_locations[0]

                    if target:
                        for un in self.units(u).idle:
                            await self.do(un.attack(target))


run_game(maps.get("RomanticideLE"), [
    Bot(Race.Protoss, ProtossBot()), Computer(Race.Terran, Difficulty.Medium)
], realtime=False)
