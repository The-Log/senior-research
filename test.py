#!/usr/bin/env python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function
import matplotlib.pyplot as plt
import h5py
from time import sleep
from vizdoom import *
import numpy as np
import cv2
game = DoomGame()

# print = lambda *args, **kwargs : None
# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../scenarios/basic.cfg")
# game.load_config("../../scenarios/simpler_basic.cfg")
# game.load_config("../../scenarios/rocket_basic.cfg")
# game.load_config("../../scenarios/deadly_corridor.cfg")
game.load_config("deathmatch.cfg")
game.set_doom_scenario_path('D3-tx_battle_99maps.wad')
# game.set_doom_scenario_path('/home/tensorpro/wads/testingnd.wad')
# game.set_sound_enabled(True)
# game.set_doom_scenario_path('/home/tensorpro/wads/o7/tech.wad')
# game.set_doom_scenario_path('/home/tensorpro/wads/3d/1.wad')
game.set_doom_map('MAP49')
# game.load_config("../../scenarios/defend_the_center.cfg")
# game.load_config("../../scenarios/defend_the_line.cfg")
# game.load_config("../../scenarios/health_gathering.cfg")
# game.load_config("../../scenarios/my_way_home.cfg")
# game.load_config("../../scenarios/predict_position.cfg")
# game.load_config("../../scenarios/take_cover.cfg")w

# Enables freelook in engine
game.add_game_args("+freelook 1"
                   "+sv_noautoaim 1 ")

def is_corner(d, i, j):
    return d[i,j] not in (d[i-1,j], d[i,j-1], d[i,j+1], d[i+1,j])

def vertices(d):
    h, w = d.shape[:2]
    vtx = []
    zs = np.zeros((h,w))
    for i in range(h-1):
        for j in range(w-1):
            if is_corner(d,i,j):
                vtx.append([i,j])
                zs[i,j]=1
    return zs

def seg(d, l):
    h, w = d.shape
    s = np.zeros_like(d)
    c=1
    prev=-1
    for i in range(h):
        for j in range(1,w):
            if l[i,j]!=0:
                s[i,j]=s[i,j-1]
            elif d[i,j]==prev:
                s[i,j]=c
            else:
                c+=1
            prev = d[i,j]
        c+=1
    return s
cv2.floodFill

#game.set_doom_map('map02')
game.set_screen_format(ScreenFormat.RGB24)
game.set_screen_resolution(ScreenResolution.RES_1920X1080)
#game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_render_crosshair(True)
# Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)
game.set_labels_buffer_enabled(True)
game.set_depth_buffer_enabled(True)
game.set_automap_buffer_enabled(True)



randomize=1
#game.send_game_command("pukename set_value always 4 %i" % randomize)



import cv2
from itertools import product

class Entity:

    def __init__(self, top, left, bottom, right, scale=1):
        self.top = top
        self.left = left
        self.right = right
        self.bottom = bottom
    def update(self, h, c):
        if h > self.bottom:
            self.bottom = h
        if h < self.top:
            self.top = h
        if c > self.right:
            self.right = c
        if c < self.left:
            self.left = c

    def __repr__(self):
        format_str = "\nBOX [\nTop left: {0}\nBottom Right: {1}]\n\n"
        return format_str.format((self.top, self.left), (self.bottom, self.right))

def draw_bbox(screen, entity, color=[0,0,255]):
    screen = screen.copy()
    cs = l,r = entity.left, entity.right
    rs = t,b = entity.top, entity.bottom
    for col in cs:
        screen[t:b,col]=color
    for row in rs:
        screen[row,l:r]=color
    return screen

def create_training_data(screen, entity, i):
    screen = screen.copy()
    cs = l,r = entity.left, entity.right
    rs = t,b = entity.top, entity.bottom
    roi = screen[t:b,l:r]
    a = (r-l) * (b-t)
    if a > 500:
        print(i, "\t", a)
    

def show_entities(screen, label_buffer, ds=1):
    screen_ds = screen[::ds, ::ds]
    screen_ds = screen
    h,w = screen.shape[:2]
    screen_ds = np.zeros((h//ds,w//ds))
    label_ds = label_buffer[::ds, ::ds]
    entities = bboxes(label_ds).values()
    i = 0
    for e in entities:
        create_training_data(screen, e, i)
        screen_ds = draw_bbox(screen_ds, e, color=255)
        i = i + 1
    boxes = cv2.resize(screen_ds, (w,h))
    y,x = np.nonzero(boxes)
    # print(y,x)
    screen_out = screen.copy()
    screen_out[y,x]=[0,255,0]
    return screen_out

def bboxes(labels_buffer):
    entities = {}
    # labels_buffer[labels_buffer==255]=0
    for r,c in zip(*np.nonzero(labels_buffer)):
        label = labels_buffer[r,c]
        if label not in entities.keys():
            entities[label] = Entity(r,c,r,c)
        else:
            entities[label].update(r,c)
    return entities

episodes=1
def segment(d, n=10):
    d = d.copy()
    
    seg = np.zeros_like(d)
    seg[:-n,:]=np.abs(d[n:,:]-d[:-n,:])>5
    # seg[d<=10]=0
    # seg[:-n,:]=seg[:-n,:]==seg[n:,:]
    return seg
# if False:
game.init()
        
def smooth(seg, n):
    disp = n//2
    return seg
    return np.where(seg[:-n-disp,:]==seg[n+disp:,:],seg[:-n,:],seg[:-disp,:])

#f1 = open('imagenumber.txt','a')
#image_number = int(f1.read())

class FieldCollector:

    def __init__(self, h5file, name, shape=[], dtype=float):
        self.shape = shape
        self.name = name
        self.f = h5file
        self.dtype = dtype
        if name not in self.f.keys():
            self.f.create_dataset(name, [0]+list(shape), dtype, maxshape=[None]+list(shape))
        self.dataset = self.f[name]
        self.count = len(self.f[name])

    def append(self, value):
        if len(self.shape)==0 or list(self.shape)==list(value.shape):
            self.dataset.resize([self.count+1]+list(self.shape))
            self.dataset[self.count, :len(value)]=value
            self.count+=1
            print(self.count)


class DoomCollector:

    def __init__(self, filename, screen_shape, max_entities=30):
        self.filename = filename
        self.f = h5py.File(filename, 'a')
        self.depths = FieldCollector(self.f, 'depth', screen_shape)
        self.screens = FieldCollector(self.f, 'screen', screen_shape)
        self.bboxes = FieldCollector(self.f, 'bboxes', [max_entities, 4])
        self.names = FieldCollector(self.f, 'labels', [max_entities], dtype='S15')
        self.max_entities=max_entities

    def add_entities(self, labels):
        bboxes = np.zeros((self.max_entities, 4))
        names = np.zeros((self.max_entities), dtype='S15')
        for idx, l in enumerate(labels):
            if idx >= self.max_entities:
                break
            bboxes[idx] = [l.y, l.x, l.width, l.height]
            names[idx] = np.string_(l.object_name)
            
        self.bboxes.append(bboxes)
        self.names.append(names)
        return names, bboxes

dc = DoomCollector('dataset.f5', [160,120])
j = 0
for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()

        #print("Player position X:", state.game_variables[0], "Y:", state.game_variables[1], "Z:", state.game_variables[2])
        


        sc = state.screen_buffer
        lb = state.labels_buffer
        if j % 5 == 0:
            dc.add_entities(state.labels)
            resized_sc =  cv2.resize(sc, (120, 160))
            dc.screens.append(resized_sc)

            d = state.depth_buffer
            resized_d =  cv2.resize(d, (120, 160) )
            dc.depths.append(resized_d)

        j = j + 1
        #cv2.imshow('labels',state.labels_buffer)
        #cv2.imshow('depth',state.depth_buffer)
        #cv2.imshow('entities',show_entities(sc,lb))
        #cv2.imwrite("training-data/training"+ str(image_number) +".png", sc)
        
        # cv2.imshow('map',state.automap_buffer)
        # cv2.imshow('seg', lel(d))
        # cv2.imshow('ff', flood_fill(d,[0,0],fill_val=200).astype(np.uint8))
        # plt.imshow(flood_fill(d,[0,0], fill_val=200))
        # plt.show()
        
        def lel(d, n=20):
            seg = segment(d,n)*200
            return smooth(seg,20).astype(np.uint8)
            return cv2.Canny(segment(d,n)*200, 100,200)
        #cv2.imshow('seg2', lel(d.T, 60).T)
        #print(np.unique(lb))
        cv2.waitKey(1)
        game.advance_action(2)
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        print("State #" + str(state.number))
        print("Game variables: ", state.game_variables)
        print("Action:", last_action)
        print("Reward:", reward)
        print("=====================")

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
cv2.destroyAllWindows()
# wr = open('imagenumber.txt', 'w')
# wr.write(str(image_number))



# def flood_fill(d, point, thresh=10, fill_arr=None, fill_val=1):
#     if fill_arr is None:
#         fill_arr = np.zeros_like(d)

#     h,w = d.shape[:2]
#     def neighbors(point):
#         disps = np.array([[-1, 1], [1, -1], [1, 1], [-1, -1]])
#         return [(y,x) for (y,x) in (disps+np.array(point))
#                 if (y > 0 and y < h and x > 0 and x < w and fill_arr[y,x]==0)]

    
#     def recur(point,s=0):
#         fill_arr[point]=fill_val
#         for n in neighbors(point):
#             print('np',n, point)
#             if abs(d[tuple(point)]-d[n])<=thresh and fill_arr[n]==0:
#                 recur(n, s+1)
#     recur(point)
#     return fill_arr
