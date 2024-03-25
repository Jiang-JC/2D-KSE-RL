import numpy as np
import random
from collections import deque
import pickle

class MemoryBuffer:

    def __init__(self, size=2000000):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.maxDistSize = 200
        self.len = 0
        self.pos = 0
        self.cont = 0
        self.t = [0]*self.maxDistSize

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def __len__(self):
        return self.len

    def add(self, s, a, r, s1, Test):
        if Test:
            return
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s,a,r,s1)
        
        self.t[self.cont%self.maxDistSize] = self.pos
        self.cont += 1
        self.pos = self.cont%self.maxSize

        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
        
    def save_buffer(self,_ep):
        with open('./Buffer/'+str(_ep)+'len.txt', "wb") as fp:   #Pickling
            pickle.dump(self.len, fp)
        with open("./Buffer/"+str(_ep)+"pos.txt", "wb") as fp:   #Pickling
            pickle.dump(self.pos, fp)
        with open("./Buffer/"+str(_ep)+"cont.txt", "wb") as fp:   #Pickling
            pickle.dump(self.cont, fp)
        with open("./Buffer/"+str(_ep)+"buffer.txt", "wb") as fp:   #Pickling
            pickle.dump(self.buffer, fp)

    def load_buffer(self,_ep):
        with open("./Buffer/"+str(_ep)+"len.txt", "rb") as fp:   # Unpickling
            self.len = int(pickle.load(fp))
        with open("./Buffer/"+str(_ep)+"pos.txt", "rb") as fp:   # Unpickling
            self.pos = int(pickle.load(fp))
        with open("./Buffer/"+str(_ep)+"cont.txt", "rb") as fp:   # Unpickling
            self.cont = int(pickle.load(fp))
        with open("./Buffer/"+str(_ep)+"buffer.txt", "rb") as fp:   # Unpickling
            self.buffer = pickle.load(fp)
