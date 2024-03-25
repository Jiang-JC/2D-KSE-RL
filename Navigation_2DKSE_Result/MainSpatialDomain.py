import numpy as np
from KS2D import KS
import torch
import train
import buffer

import imageio
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #use GPU or CPU

MAX_STEPS = 1001             #total steps in one epoch
N_SAMPLE = 64                #number of sample points
S_DIM = 256                  #number of equispaced sensors
A_DIM = 36                   #number of equispaced actuators
A_MAX = 3                    #maximum amplitude for the actuation [-A_MAX, A_MAX]

ks = KS(L = 20, N = N_SAMPLE, a_dim = A_DIM)  #Kuramoto-Sivashinsky class initialization

print('State Dimensions :- ', S_DIM)
print('Action Dimensions :- ', A_DIM)
print('Action Max :- ', A_MAX)

Plot = True
Save = True
Test = False                 #Test mode: load and use the policy w/o optimization

init_point = np.loadtxt('./FixedPoints/1fixed_point.dat')
target_point = np.loadtxt('./FixedPoints/2fixed_point.dat')

if Plot or Save:
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

ram = buffer.MemoryBuffer()               #memory class initialization

# initalize new trainer
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device, Test)     #RL class initialization 

trainer.load_best_models()
trainer.noise_size = 0

new_observation = init_point                          #load initial condition

with imageio.get_writer('./Figure/testModel_spatialDomain.gif', mode='I',fps=5) as writer:
    for r in range(MAX_STEPS):

        state = np.float32((new_observation[0::4, 0::4]).flatten())
        observation = new_observation
        action = trainer.get_action(state, Test=Test)
        new_observation = ks.advance(observation,action)

        distance = - np.linalg.norm(new_observation - target_point)

        # Plot Part
        if r % 5 == 0 and Plot == True:
            ax1.cla()
            ax1.contourf(ks.X, ks.Y , new_observation)

            ax1.set_xticks([0, 5, 10, 15, 20])
            ax1.set_yticks([0, 5, 10, 15, 20])

            ax2.cla()
            ax2.contourf(ks.X, ks.Y , target_point)

            ax2.set_xticks([0, 5, 10, 15, 20])
            ax2.set_yticks([0, 5, 10, 15, 20])

            plt.pause(0.01)
            plt.show(block=False)

        if Save and r % 5 == 0:
            ax1.cla()
            ax1.contourf(ks.X, ks.Y , new_observation)

            ax1.set_xticks([0, 5, 10, 15, 20])
            ax1.set_yticks([0, 5, 10, 15, 20])

            ax2.cla()
            ax2.contourf(ks.X, ks.Y , target_point)
            
            ax2.set_xticks([0, 5, 10, 15, 20])
            ax2.set_yticks([0, 5, 10, 15, 20])
            
            plt.savefig('foo.png')
            
            image = imageio.v2.imread('foo.png')
            writer.append_data(image)

            