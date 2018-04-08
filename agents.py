import numpy as np
from vizdoom import *
import itertools as it
import random 

buttons = [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_BACKWARD, Button.MOVE_FORWARD, Button.TURN_RIGHT, Button.TURN_LEFT, Button.TURN180, Button.STRAFE, Button.ATTACK ]
common_enemies = ['Zombieman', 'ShotgunGuy', 'Archvile', 'Revenant', 'RevenantTracer', 'Fatso', 'ChaingunGuy', 'DoomImp', 'Demon', 'Spectre', 'Cacodemon', 'BaronOfHell', 'BaronBall', 'HellKnight', 'LostSoul', 'SpiderMastermind', 'Arachnotron', 'Cyberdemon', 'PainElemental', 'WolfensteinSS', 'CommanderKeen', 'BossBrain', 'BossEye', 'BossTarget']

def get_actions():
    return [[1 if i == j else 0 for j in buttons] for i in buttons]

class BaseAgent(object):
    def __init__(self, game, actions):
        self.env = game
        self.actions = actions

    def aim_at(self, entity, state):
        db = state.depth_buffer
        center = (self.env.get_screen_width() / 2, self.env.get_screen_height() / 2)
        distance = center[0] - int(entity[1])
        y = int(center[1]) + 20
        env_w = self.env.get_screen_width()
        # print(distance)
        if abs(distance) < 100:
            if int(center[0]) > int(entity[1]):
                if(db[y, 20] <=5):
                    self.env.make_action(self.actions[5])
                else:
                    self.env.make_action(self.actions[0])
            elif center[0] < int(entity[1]):
                if(db[y, env_w - 20] <=5):
                    self.env.make_action(self.actions[4])
                else:
                    self.env.make_action(self.actions[1])
        else:
            if int(center[0]) > int(entity[1]):
                self.env.make_action(self.actions[5])
            elif center[0] < int(entity[1]):
                self.env.make_action(self.actions[4])

    def action(self, state):
        n = state.number
        v = state.game_variables
        sb = state.screen_buffer
        db = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        center = (self.env.get_screen_width() / 2, self.env.get_screen_height() / 2)
        
        entities = []
        for e in state.labels:
            if (e.object_name != 'DoomPlayer'):
                entities.append((e.object_name, (2 * e.x + e.width) / 2, (2 * e.y + e.height)/2 , e.width, e.height))
                
        ammo = state.game_variables[0]
        health = state.game_variables[1]
        enemies = [i for i in entities if i[0] in common_enemies]

        env_med = [i for i in entities if i[0] in ['CustomMedikit']]
        env_ammo = [i for i in entities if i[0] in ['Clip']]
        if health < 35 and len(env_med) > 0 and len(enemies) <= 1:
            kit = env_med[-1]
            self.aim_at(kit, state)
            self.env.make_action(self.actions[3])
        elif len(enemies) > 0 and ammo > 0:
            enemy = enemies[-1]
            distance = center[0] - int(enemy[1])
            if (abs(distance) < int(enemy[3]) / 4):
                self.env.make_action(self.actions[8])
            else:
                self.aim_at(enemy, state)
        elif ammo < 5 and len(env_ammo) > 0:
            print(env_ammo)
            a = env_ammo[-1]
            self.aim_at(a, state)
            self.env.make_action(self.actions[3])
        else:
            x = int(center[0])
            y = int(center[1]) + 20
            cd = db[y, x]
            if (db[y, x] <= 5 and db[y, x + 50] <=5 and db[y, x - 50] <=5 ):
                self.env.make_action(self.actions[6])
            elif(db[y,x] <=5 and db[y, x - 50] <=5):
                self.env.make_action(self.actions[5])
            elif(db[y,x] <=5 and db[y, x + 50] <=5):
                self.env.make_action(self.actions[4])
            self.env.make_action(self.actions[3])


class KeyBoardAgent(BaseAgent):
    def __init__(self, game, actions):
        self.env = game
        self.actions = actions

    def action(self, state):
        """
        actions[0] = MOVE_LEFT, actions[1] = MOVE_RIGHT
        actions[2] = MOVE_BACKWARD, actions[3] = MOVE_FORWARD
        actions[4] = TURN_RIGHT, actions[4] = TURN_LEFT
        """
        key = input('')
        if key == 'a':
            print("pressed a")
            self.env.make_action(self.actions[0])
        if key == 'd':
            print("pressed d")
            self.env.make_action(self.actions[1])
        if key == 's':
            print("pressed s")
            self.env.make_action(self.actions[2])
        if key == 'w':
            print("pressed w")
            self.env.make_action(self.actions[3])
        if key == '4':
            print("pressed 4")
            self.env.make_action(self.actions[4])
        if key == '5':
            print("pressed 5")
            self.env.make_action(self.actions[5])
        if key == '6':
            print("pressed 6")
            self.env.make_action(self.actions[6])
        if key == '7':
            print("pressed 7")
            self.env.make_action(self.actions[7])
        if key == '8':
            print("pressed 8")
            self.env.make_action(self.actions[8])
        
        else:
            return
