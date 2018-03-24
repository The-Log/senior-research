import numpy as np
from vizdoom import *
import itertools as it
from random import choice

buttons = [Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_BACKWARD, Button.MOVE_FORWARD, Button.ATTACK, Button.TURN_RIGHT, Button.TURN_LEFT]
common_enemies = ['Zombieman', 'ShotgunGuy', 'Archvile', 'Revenant', 'RevenantTracer', 'Fatso', 'ChaingunGuy', 'DoomImp', 'Demon', 'Spectre', 'Cacodemon', 'BaronOfHell', 'BaronBall', 'HellKnight', 'LostSoul', 'SpiderMastermind', 'Arachnotron', 'Cyberdemon', 'PainElemental', 'WolfensteinSS', 'CommanderKeen', 'BossBrain', 'BossEye', 'BossTarget']

def get_actions():
    return [[1 if i == j else 0 for j in buttons] for i in buttons]

class BaseAgent(object):
    def __init__(self, game, actions):
        self.env = game
        self.actions = actions
        
    def action(self, state):
        n = state.number
        vars = state.game_variables
        sb = state.screen_buffer
        db = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        center = (self.env.get_screen_width() / 2, self.env.get_screen_height() / 2)
        
        entities = []
        for e in state.labels:
            if (e.object_name != 'DoomPlayer'):
                entities.append((e.object_name, (2 * e.x + e.width) / 2, (2 * e.y + e.height)/2 ))
                
        enemies = [i for i in entities if i[0] in common_enemies] 
        
        if len(enemies) > 0:
            enemy = enemies[0]
            print(center, '\n', enemy)
            if int(center[0]) > int(enemy[1]):
                self.env.make_action(self.actions[0])
            elif int(center[0]) < int(enemy[1]):
                self.env.make_action(self.actions[1])
            else:
                self.env.make_action(self.actions[4])

class KeyBoardAgent(BaseAgent):
    def __init__(self, game, actions):
        self.env = game
        self.actions = actions

    def action(self, state):
        """
        actions[0] = MOVE_LEFT, actions[1] = MOVE_RIGHT
        actions[2] = MOVE_BACKWARD, actions[3] = MOVE_FORWARD
        actions[4] = SHOOT
        actions[5] = TURN_RIGHT, actions[6] = TURN_LEFT
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
        
        else:
            return
