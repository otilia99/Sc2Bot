import os
import sc2
import cv2
import random
import numpy as np
from enum import Enum
from sc2.units import Units
from sc2.player import Bot, Computer
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.constants import UnitTypeId, AbilityId, UpgradeId, BuffId

os.environ["SC2PATH"] = 'D:/Downloads/Starcraft II/Starcraft II'


class Scenario(Enum):
    PEACE = 0
    BASE_ATTACK = 1
    OUTER_BASES_ATTACK = 2
    ATTACK_ENEMY_BASE = 3
    FIGHT_BACK_ATTACK = 4


class ProtossBot(sc2.BotAI):
    def __init__(self):
        super().__init__()
        self.IT_PER_MIN = 165
        self.MAX_WORKERS = 80
        self.do_smth_after_this = 0
        self.train_data = []
        self.scouting_done = False
        self.point = None
        self.ramp_pos = None
        self.main_base = None
        self.current_scenario = Scenario.PEACE
        self.late_game = False
        self.no_of_buildings = 0
        self.prev_no_of_buildings = 0
        self.attack_units = {
            UnitTypeId.STALKER: [15, 5],
            UnitTypeId.VOIDRAY: [8, 3],
            UnitTypeId.PHOENIX: [8, 2],
            UnitTypeId.SENTRY: [4, 3],
            UnitTypeId.ZEALOT: [6, 5],
            UnitTypeId.ORACLE: [1, 1],
            UnitTypeId.COLOSSUS: [5, 4]}

        self.all_unit_names = {UnitTypeId.STALKER: "stalker",
                               UnitTypeId.VOIDRAY: "voidray",
                               UnitTypeId.PHOENIX: "phoenix",
                               UnitTypeId.SENTRY: "sentry",
                               UnitTypeId.ZEALOT: "zealot",
                               UnitTypeId.ORACLE: "oracle",
                               UnitTypeId.COLOSSUS: "colossus"}

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.upgrades()
        await self.build_offensive_force()
        await self.scout()
        await self.rally_new_building()
        await self.log_army_state()
        await self.apply_chronoboost()
        # await self.draw_base()
        await self.attack()

        if iteration == 0:
            await self.chat_send("(pylon) hi (glhf)")

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
        supply_limit = 10 if self.late_game else 5
        if self.supply_left < supply_limit and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first, placement_step=6)

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

    async def expand_correct_location(self):
        location = await self.get_next_expansion()
        await self.build(UnitTypeId.NEXUS, near=location, random_alternative=False)

    async def expand(self):
        if self.alive_or_pending_units(UnitTypeId.NEXUS) < 3 and \
                self.can_afford(UnitTypeId.NEXUS):
            await self.expand_correct_location()
            return

        if self.alive_or_pending_units(UnitTypeId.NEXUS) < 5 and self.can_afford(UnitTypeId.NEXUS) and \
                self.alive_or_pending_units(UnitTypeId.STALKER) >= 8 and \
                self.alive_or_pending_units(UnitTypeId.ZEALOT) >= 3 and \
                self.alive_or_pending_units(UnitTypeId.VOIDRAY) >= 6:
            await self.expand_correct_location()
            return

        if self.alive_or_pending_units(UnitTypeId.NEXUS) < 6 and self.can_afford(UnitTypeId.NEXUS) and \
                self.units(UnitTypeId.STALKER).amount > 10 and self.units(UnitTypeId.VOIDRAY).amount > 8:
            await self.expand_correct_location()

    async def production(self):
        if self.minerals > 500:
            for gw in self.units(UnitTypeId.GATEWAY).ready:
                await self.train_units(UnitTypeId.STALKER, gw)
                await self.train_units(UnitTypeId.ZEALOT, gw)

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

    def random_location_variance(self, enemy_start_location, default_range=20):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-default_range, default_range)) / 100) * enemy_start_location[0]
        y += ((random.randrange(-default_range, default_range)) / 100) * enemy_start_location[1]

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

    def interpolate_location(self, base_location, current_location, t):
        x0 = base_location[0]
        y0 = base_location[1]

        x1 = current_location[0]
        y1 = current_location[1]

        x = x1 + (x0 - x1) * t
        y = y1 + (y0 - y1) * t

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

    def structure_status(self, unit):
        return self.units(unit).ready.exists or self.already_pending(unit)

    async def apply_chronoboost(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            for gw in self.units(UnitTypeId.GATEWAY).ready:
                if not gw.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, gw))
                    return

    async def offensive_force_buildings(self):
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random

            if not (self.structure_status(UnitTypeId.GATEWAY) and self.structure_status(UnitTypeId.CYBERNETICSCORE)
                    and self.structure_status(UnitTypeId.STARGATE) and self.structure_status(UnitTypeId.FORGE)):
                if not self.structure_status(UnitTypeId.GATEWAY):
                    if self.can_afford(UnitTypeId.GATEWAY):
                        await self.build(UnitTypeId.GATEWAY, near=pylon, placement_step=1)
                    return
                if not self.structure_status(UnitTypeId.CYBERNETICSCORE):
                    if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                        await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon, placement_step=1)
                    return
                if self.alive_or_pending_units(UnitTypeId.STALKER) < 3 and \
                        self.alive_or_pending_units(UnitTypeId.ZEALOT) < 2:
                    return
                if not self.structure_status(UnitTypeId.STARGATE):
                    if self.can_afford(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon, placement_step=1)
                    return
                if self.alive_or_pending_units(UnitTypeId.STALKER) < 5 and \
                        self.alive_or_pending_units(UnitTypeId.ZEALOT) < 4:
                    return
                if not self.structure_status(UnitTypeId.FORGE):
                    if self.can_afford(UnitTypeId.FORGE):
                        await self.build(UnitTypeId.FORGE, near=pylon, placement_step=2)
                    return

            if self.alive_or_pending_units(UnitTypeId.STALKER) < 7 and \
                    self.alive_or_pending_units(UnitTypeId.ZEALOT) < 6 and \
                    self.alive_or_pending_units(UnitTypeId.VOIDRAY) < 3:
                return

            if self.units(UnitTypeId.GATEWAY).amount < 3:
                if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon, placement_step=2)

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if self.units(UnitTypeId.STARGATE).amount < 3:
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon, placement_step=2)

                if self.units(UnitTypeId.ROBOTICSFACILITY).amount < 1 and \
                        self.can_afford(UnitTypeId.ROBOTICSFACILITY) and \
                        not self.already_pending(UnitTypeId.ROBOTICSFACILITY):
                    await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon, placement_step=3)

                if self.units(UnitTypeId.ROBOTICSFACILITY).ready.exists and self.can_afford(UnitTypeId.ROBOTICSBAY) \
                        and not self.already_pending(UnitTypeId.ROBOTICSBAY) \
                        and self.units(UnitTypeId.ROBOTICSBAY).amount < 1:
                    await self.build(UnitTypeId.ROBOTICSBAY, near=pylon, placement_step=3)

                if self.units(UnitTypeId.TWILIGHTCOUNCIL).amount < 1:
                    if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and \
                            not self.already_pending(UnitTypeId.TWILIGHTCOUNCIL):
                        await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon, placement_step=2)

                if self.units(UnitTypeId.TWILIGHTCOUNCIL).ready.exists \
                        and self.units(UnitTypeId.TEMPLARARCHIVE).amount < 1:
                    if self.can_afford(UnitTypeId.TEMPLARARCHIVE) and \
                            not self.already_pending(UnitTypeId.TEMPLARARCHIVE):
                        await self.build(UnitTypeId.TEMPLARARCHIVE, near=pylon, placement_step=2)

                if self.units(UnitTypeId.STARGATE).ready.exists and self.units(UnitTypeId.VOIDRAY).amount > 7:
                    if self.can_afford(UnitTypeId.FLEETBEACON) and not self.already_pending(UnitTypeId.FLEETBEACON):
                        await self.build(UnitTypeId.FLEETBEACON, near=pylon, placement_step=2)

    async def do_upgrade(self, ability_id, upgrade_id, building, unit_id, unit_amount):
        if self.can_afford(ability_id) and not self.already_pending_upgrade(upgrade_id) and \
                self.units(unit_id).amount >= unit_amount:
            await self.do(building.research(upgrade_id))

    async def upgrades(self):
        if self.units(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
            tcouncil = self.units(UnitTypeId.TWILIGHTCOUNCIL).ready.first
            if self.can_afford(AbilityId.RESEARCH_CHARGE) and not self.already_pending_upgrade(UpgradeId.CHARGE):
                await self.do(tcouncil.research(UpgradeId.CHARGE))

            if self.can_afford(AbilityId.RESEARCH_BLINK) and not self.already_pending_upgrade(UpgradeId.BLINKTECH):
                await self.do(tcouncil.research(UpgradeId.BLINKTECH))

        if self.units(UnitTypeId.FORGE).ready.exists:
            built_forge = self.units(UnitTypeId.FORGE).ready.first
            await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1,
                                  UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1, built_forge,
                                  UnitTypeId.STALKER, 5)

            await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1,
                                  UpgradeId.PROTOSSGROUNDARMORSLEVEL1, built_forge,
                                  UnitTypeId.STALKER, 10)

            await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1,
                                  UpgradeId.PROTOSSSHIELDSLEVEL1, built_forge,
                                  UnitTypeId.STALKER, 15)

            if self.units(UnitTypeId.TWILIGHTCOUNCIL).ready.exists:
                await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2,
                                      UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2, built_forge,
                                      UnitTypeId.VOIDRAY, 5)

                await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2,
                                      UpgradeId.PROTOSSGROUNDARMORSLEVEL2, built_forge,
                                      UnitTypeId.VOIDRAY, 7)

                await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2,
                                      UpgradeId.PROTOSSSHIELDSLEVEL1, built_forge,
                                      UnitTypeId.VOIDRAY, 9)

                await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3,
                                      UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3, built_forge,
                                      UnitTypeId.VOIDRAY, 5)

                await self.do_upgrade(AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3,
                                      UpgradeId.PROTOSSGROUNDARMORSLEVEL3, built_forge,
                                      UnitTypeId.VOIDRAY, 7)

            if self.units(UnitTypeId.TEMPLARARCHIVE).ready.exists:
                temparch = self.units(UnitTypeId.TEMPLARARCHIVE).ready.first
                await self.do_upgrade(AbilityId.TWILIGHTCOUNCILRESEARCH_RESEARCHPSIONICSURGE,
                                      UpgradeId.PSISTORMTECH, temparch,
                                      UnitTypeId.HIGHTEMPLAR, 2)

            if self.units(UnitTypeId.ROBOTICSBAY).ready.exists:
                robbay = self.units(UnitTypeId.ROBOTICSBAY).ready.first
                await self.do_upgrade(AbilityId.RESEARCH_EXTENDEDTHERMALLANCE, UpgradeId.EXTENDEDTHERMALLANCE, robbay,
                                      UnitTypeId.COLOSSUS, 2)

            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                cybercore = self.units(UnitTypeId.CYBERNETICSCORE).ready.first
                await self.do_upgrade(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1,
                                      UpgradeId.PROTOSSAIRWEAPONSLEVEL1, cybercore,
                                      UnitTypeId.VOIDRAY, 5)

                await self.do_upgrade(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1,
                                      UpgradeId.PROTOSSAIRARMORSLEVEL1, cybercore,
                                      UnitTypeId.VOIDRAY, 15)

                await self.do_upgrade(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2,
                                      UpgradeId.PROTOSSAIRWEAPONSLEVEL2, cybercore,
                                      UnitTypeId.VOIDRAY, 8)

                await self.do_upgrade(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3,
                                      UpgradeId.PROTOSSAIRWEAPONSLEVEL3, cybercore,
                                      UnitTypeId.VOIDRAY, 10)

    @property
    def ret_pos(self):
        if self.ramp_pos is None:
            self.ramp_pos = list(self.main_base_ramp.upper)[0]
        if self.point is None:
            self.point = self.random_location_variance(self.ramp_pos, default_range=10)
        return self.point, self.ramp_pos

    def alive_or_pending_units(self, unit_id):
        return self.units(unit_id).ready.amount + self.already_pending(unit_id)

    async def train_units(self, unit_id, building, unit_cond_id=None, amount=None):
        if unit_cond_id is None and amount is None:
            cond = True
        else:
            cond = self.units(unit_cond_id).amount >= amount
        if self.can_afford(unit_id) and self.supply_left > 0 and cond:
            await self.do(building.train(unit_id))

    async def rally_units(self):
        rally_buildings = [UnitTypeId.GATEWAY, UnitTypeId.STARGATE, UnitTypeId.ROBOTICSFACILITY]
        location_point, _ = self.ret_pos
        for building in rally_buildings:
            for b in self.units(building):
                await self.do(b(AbilityId.RALLY_BUILDING, location_point))

    async def rally_new_building(self):
        rally_buildings = [UnitTypeId.GATEWAY, UnitTypeId.STARGATE, UnitTypeId.ROBOTICSFACILITY]
        self.no_of_buildings = sum([self.units(building).ready.amount for building in rally_buildings])
        if self.no_of_buildings != self.prev_no_of_buildings:
            await self.rally_units()
        self.prev_no_of_buildings = self.no_of_buildings

    async def build_offensive_force(self):
        await self.production()
        if self.alive_or_pending_units(UnitTypeId.ZEALOT) < 2:
            for gw in self.units(UnitTypeId.GATEWAY).ready:
                await self.train_units(UnitTypeId.ZEALOT, gw)
            return

        if self.alive_or_pending_units(UnitTypeId.STALKER) < 3:
            for gw in self.units(UnitTypeId.GATEWAY).ready:
                await self.train_units(UnitTypeId.STALKER, gw)
            return

        if self.units(UnitTypeId.STALKER).amount >= 3 and self.units(UnitTypeId.VOIDRAY).amount \
                + self.already_pending(UnitTypeId.VOIDRAY) < 5:
            for sg in self.units(UnitTypeId.STARGATE).ready.idle:
                await self.train_units(UnitTypeId.VOIDRAY, sg)
            return

        for sg in self.units(UnitTypeId.STARGATE).ready.idle:
            await self.train_units(UnitTypeId.VOIDRAY, sg)
            await self.train_units(UnitTypeId.PHOENIX, sg, unit_cond_id=UnitTypeId.VOIDRAY, amount=6)
            if self.units(UnitTypeId.ORACLE).amount < 1:
                await self.train_units(UnitTypeId.ORACLE, sg)
            # await self.train_units(UnitTypeId.CARRIER, sg, unit_cond_id=UnitTypeId.VOIDRAY, amount=7)
            await self.train_units(UnitTypeId.TEMPEST, sg, unit_cond_id=UnitTypeId.VOIDRAY, amount=6)

        for gw in self.units(UnitTypeId.GATEWAY).ready.idle:
            await self.train_units(UnitTypeId.STALKER, gw)
            await self.train_units(UnitTypeId.SENTRY, gw, unit_cond_id=UnitTypeId.VOIDRAY, amount=3)
            await self.train_units(UnitTypeId.ZEALOT, gw, unit_cond_id=UnitTypeId.STALKER, amount=1)

        for rb in self.units(UnitTypeId.ROBOTICSBAY).ready.idle:
            await self.train_units(UnitTypeId.COLOSSUS, rb, unit_cond_id=UnitTypeId.VOIDRAY, amount=7)

        for ta in self.units(UnitTypeId.GATEWAY).ready.idle:
            await self.train_units(UnitTypeId.HIGHTEMPLAR, ta, UnitTypeId.VOIDRAY, 7)

        for nex in self.units(UnitTypeId.NEXUS).ready.idle:
            if self.units(UnitTypeId.MOTHERSHIP).amount < 1:
                await self.train_units(UnitTypeId.MOTHERSHIP, nex, unit_cond_id=UnitTypeId.TEMPEST, amount=1)

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def log_army_state(self):
        print(self.current_scenario.name)

    async def attack_target(self, unit_id_list, target=None):
        for unit_id in unit_id_list:
            for unit in self.units(unit_id):
                if target is None and self.known_enemy_units:
                    target = self.known_enemy_units.closest_to(unit)
                await self.do(unit.attack(target))

    @property
    def get_main_base(self):
        if self.main_base is None:
            self.main_base = self.units(UnitTypeId.NEXUS).first
        return self.main_base

    def nexus_index(self):
        first_base = self.get_main_base
        return enumerate(self.units(UnitTypeId.NEXUS).sorted_by_distance_to(first_base))

    def nexus_is_under_attack(self):
        for index, nexus in self.nexus_index():
            close_enemies = self.known_enemy_units.closer_than(10, nexus)
            if close_enemies:
                return index, close_enemies
        return None, []

    def retreat_distance(self, unit):
        return unit.distance_to(self.get_main_base) < 70

    def handle_scenario(self):
        all_attacking_enemies = []
        for unit in self.units.owned:
            close_enemies = self.known_enemy_units.closer_than(10, unit)
            if close_enemies:
                all_attacking_enemies.extend(close_enemies)
        index, close_enemies = self.nexus_is_under_attack()
        if all_attacking_enemies:
            if index is not None:
                if index > 2:
                    self.current_scenario = Scenario.OUTER_BASES_ATTACK
                else:
                    self.current_scenario = Scenario.BASE_ATTACK
            else:
                if self.retreat_distance(Units(all_attacking_enemies).closest_to(self.get_main_base)):
                    self.current_scenario = Scenario.FIGHT_BACK_ATTACK
        if not self.current_scenario in [Scenario.BASE_ATTACK, Scenario.OUTER_BASES_ATTACK,
                                         Scenario.FIGHT_BACK_ATTACK] and \
                self.units(UnitTypeId.VOIDRAY).amount > 7 and self.units(UnitTypeId.STALKER).amount > 8 or \
                self.supply_army > 85:
            self.current_scenario = Scenario.ATTACK_ENEMY_BASE
            self.late_game = True
        if not self.current_scenario == Scenario.ATTACK_ENEMY_BASE and not all_attacking_enemies:
            self.current_scenario = Scenario.PEACE
        return all_attacking_enemies

    def owned_army(self):
        army = []
        for unit_id in self.attack_units.keys():
            for unit in self.units(unit_id):
                army.append(unit)
        return army

    async def retreat_all_units_if_too_far(self):
        for unit in self.owned_army():
            if not self.retreat_distance(unit):
                await self.do(unit.move(self.ret_pos[0]))

    async def retreat_if_low_shield(self, units, all_in):
        for unit in units:
            if unit.shield_percentage < 0.2 and not all_in:
                await self.do(unit.move(self.get_main_base))
            elif unit.shield_percentage < 0.6:
                retreat_pos = self.interpolate_location(self.get_main_base.position, unit.position, 0.2)
                await self.do(unit.move(retreat_pos))

    async def attack(self):
        army = self.owned_army()
        if army:
            await self.retreat_if_low_shield(army, all_in=self.current_scenario == Scenario.ATTACK_ENEMY_BASE)
        attacking_enemies = self.handle_scenario()
        index, close_enemies = self.nexus_is_under_attack()

        # if not self.current_scenario == Scenario.ATTACK_ENEMY_BASE:
        #     await self.retreat_all_units_if_too_far()

        if self.current_scenario == Scenario.PEACE:
            return
        elif self.current_scenario == Scenario.BASE_ATTACK:
            if close_enemies:
                target = close_enemies[0]
                await self.attack_target(self.attack_units.keys(), target)
        elif self.current_scenario == Scenario.OUTER_BASES_ATTACK:
            if close_enemies:
                target = close_enemies[0]
                await self.attack_target(self.attack_units.keys(), target)
        elif self.current_scenario == Scenario.FIGHT_BACK_ATTACK:
            await self.attack_target(self.attack_units.keys())
        elif self.current_scenario == Scenario.ATTACK_ENEMY_BASE:
            target = self.enemy_start_locations[0]
            await self.attack_target(self.attack_units.keys(), target)

        # if self.units(UnitTypeId.STALKER).amount > 5 and self.units(UnitTypeId.VOIDRAY).amount > 4:
        #     target = self.find_target(self.state)
        #     await self.attack_target(self.attack_units.keys(), target)

        # elif self.units(UnitTypeId.STALKER).amount >= 1 and self.units(UnitTypeId.VOIDRAY).amount >= 0:
        #     if len(self.known_enemy_units) > 0:
        #         await self.attack_target(self.attack_units.keys())


run_game(maps.get("RomanticideLE"), [
    Bot(Race.Protoss, ProtossBot()), Computer(Race.Terran, Difficulty.Medium)
], realtime=False)
