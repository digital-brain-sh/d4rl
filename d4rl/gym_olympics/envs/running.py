import random
from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import time
import pygame
import json
import sys
import os


class running(OlympicsBase):
    def __init__(self, map_id=None, seed=100, vis=200, vis_clear=5, agent1_color='light red',
                 agent2_color='blue'):
        self.maps_path = os.path.join(os.path.dirname(__file__), 'assets/maps.json')
        if map_id is None:
            map_id = random.randint(1, 11)
        Gamemap, map_index = self.choose_a_map(idx=map_id)
        # fixme(yan): penatration in some maps, need to check engine, vis
        if vis is not None:
            for a in Gamemap['agents']:
                a.visibility = vis
                a.visibility_clear = vis_clear
                if a.color == 'purple':
                    a.color = agent1_color
                    a.original_color = agent1_color
                elif a.color == 'green':
                    a.color = agent2_color
                    a.original_color = agent2_color

        self.map_index = map_index

        super(running, self).__init__(Gamemap, seed)

        self.game_name = 'running-competition'
        self.meta_map = self.create_scenario(self.game_name)
        self.original_tau = self.meta_map['env_cfg']['tau']
        self.original_gamma = self.meta_map['env_cfg']['gamma']
        self.wall_restitution = self.meta_map['env_cfg']['wall_restitution']
        self.circle_restitution = self.meta_map['env_cfg']['circle_restitution']
        self.max_step = self.meta_map['env_cfg']['max_step']
        self.energy_recover_rate = self.meta_map['env_cfg']['energy_recover_rate']
        self.speed_cap = self.meta_map['env_cfg']['speed_cap']
        self.faster = self.meta_map['env_cfg']['faster']

        self.tau = self.original_tau * self.faster
        self.gamma = 1 - (1 - self.original_gamma) * self.faster

        # self.gamma = 1  # v衰减系数
        # self.restitution = 0.5
        # self.print_log = False
        # self.print_log2 = False
        # self.tau = 0.1
        #
        # self.speed_cap =  100
        #
        # self.draw_obs = True
        # self.show_traj = True

    @staticmethod
    def reset_map(map_id, seed=100, vis=200, vis_clear=5, agent1_color='light red',
                  agent2_color='blue'):
        return running(map_id, seed, vis=vis, vis_clear=vis_clear, agent1_color=agent1_color,
                       agent2_color=agent2_color)

    def choose_a_map(self, idx=None):
        if idx is None:
            idx = random.randint(1, 4)
        MapStats = self.create_scenario("map" + str(idx), file_path=self.maps_path)
        return MapStats, idx

    def check_overlap(self):
        # todo
        pass

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 1.

        return agent_reward

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False

    def step(self, actions_list):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        # print('stepPhysics time = ', time2 - time1)
        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward = self.get_reward()
        done = self.is_terminal()

        time3 = time.time()
        obs_next = self.get_obs()
        time4 = time.time()
        # print('render time = ', time4-time3)
        # obs_next = 1
        # self.check_overlap()
        self.change_inner_state()

        return obs_next, step_reward, done, ''

    def check_win(self):
        if self.agent_list[0].finished and not (self.agent_list[1].finished):
            return '0'
        elif not (self.agent_list[0].finished) and self.agent_list[1].finished:
            return '1'
        else:
            return '-1'

    def render(self, info=None):

        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode = True

        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)

        if self.draw_obs:
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=10, gap=100)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)

        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()

    def create_scenario(self, scenario_name, file_path=None):
        module = __import__("objects")
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), 'assets/scenario.json')
        with open(file_path) as f:
            conf = json.load(f)[scenario_name]

        GameMap = dict()
        GameMap["objects"] = list()
        GameMap["agents"] = list()
        GameMap["view"] = conf["view"]

        for type in conf:
            if type == 'env_cfg':
                env_cfg_dict = conf[type]
                GameMap["env_cfg"] = env_cfg_dict
            elif type == 'obs_cfg':
                obs_cfg_dict = conf[type]
                GameMap["obs_cfg"] = obs_cfg_dict

            elif (type == "wall") or (type == "cross"):
                # print("!!", conf[type]["objects"])
                for key, value in conf[type]["objects"].items():
                    GameMap["objects"].append(getattr(module, type.capitalize())
                        (
                        init_pos=value["initial_position"],
                        length=None,
                        color=value["color"],
                        ball_can_pass=value['ball_pass'] if ("ball_pass" in value.keys()
                                                             and value['ball_pass'] == "True") else False,
                        width=value['width'] if ('width' in value.keys()) else None
                    )
                    )
            elif type == 'arc':
                for key, value in conf[type]['objects'].items():
                    # print("passable = ", bool(value['passable']))
                    GameMap['objects'].append(getattr(module, type.capitalize())(
                        init_pos=value["initial_position"],
                        start_radian=value["start_radian"],
                        end_radian=value["end_radian"],
                        passable=True if value["passable"] == "True" else False,
                        color=value['color'],
                        collision_mode=value['collision_mode'],
                        width=value['width'] if ("width" in value.keys()) else None
                    ))

            elif type in ["agent", "ball"]:
                for key, value in conf[type]["objects"].items():
                    GameMap["agents"].append(getattr(module, type.capitalize())
                        (
                        mass=value["mass"],
                        r=value["radius"],
                        position=value["initial_position"],
                        color=value["color"],
                        vis=value["vis"] if ("vis" in value.keys()) else None,
                        vis_clear=value["vis_clear"] if ("vis_clear" in value.keys()) else None
                    ),
                    )
        return GameMap


if __name__ == '__main__':
    running = running()
    map = running.choose_a_map()
    print(map)
