import h5py
import torch.utils.data as data
import numpy as np

DOOMCLASSES = {b'Clip', b'DoomImp', b'Column', b'DeadExplosiveBa', b'Blood', b'DoomPlayer', b'BulletPuff', b'ShortGreenTorch', b'DoomImpBall', b'BurningBarrel', b'CustomMedikit', b'ExplosiveBarrel'}


dick = dict(zip(DOOMCLASSES, range(len(DOOMCLASSES))))

def corner_to_center(bb):
    x1,y1,x2,y2 = bb
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    dx = (x2-x1)
    dy = (y2-y1)
    return np.array([xc,yc, dx,dy])

def scale_vals(bb, h, w):
    bb = bb.astype(np.float32)
    bb[::2]*=w
    bb[1::2]*=h
    return (bb)

class DoomDetection(data.Dataset):
    def __init__(self, f = h5py.File('dataset.h5', 'r'), dataset_name='DOOM-TS'):
        self.f = f
        self.names = f['labels']
        self.bboxes = f['bboxes']
        self.screens = f['screen']
        self.size = f['screen'].shape[0]
        self.name = dataset_name
        self.labels = []

        #slow af
        for i in range(len(self.names)):
            #xd = len(self.names[i])
            l = np.zeros(50*5)
            for j in range(len(self.names[i])):
                if(self.names[i][j] != b''):
                    n = self.names[i][j]
                    l[5*j] = dick[n]
                    y, x = self.bboxes[i][j][0], self.bboxes[i][j][1]
                    w, h = self.bboxes[i][j][2], self.bboxes[i][j][3]
                    corner = np.array([x,y,x+w,y+h])
                    center = corner_to_center(corner)
                    l[5*j+1:5*j+5] = center
            self.labels.append(l)
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        assert(i < len(self))
        return self.screens[i], self.labels[i]
    
if __name__ == '__main__':
    d = DoomDetection()
    s = set()
    for i in d.names:
        for j in i:
            s.add(j)
    print(d[0])
    #print(len(d))
