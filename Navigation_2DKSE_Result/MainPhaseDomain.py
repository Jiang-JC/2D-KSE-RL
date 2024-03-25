import numpy as np
from KS2D import KS
import torch

import train
import buffer

import imageio
import matplotlib.pyplot as plt

from PSD import PSD


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

if Save:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

ram = buffer.MemoryBuffer()               #memory class initialization

# initalize new trainer
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device, Test)     #RL class initialization 

trainer.load_best_models()

trainer.noise_size = 0

new_observation = init_point                          #load initial condition

if Save:
    ps = PSD(init_point)
    ps = np.vstack([ps, PSD(target_point)])

    plt_KS = np.array([new_observation] * 10)
    ps_KS = PSD(plt_KS[0])
    for i in range(1, plt_KS.shape[0]):
        ps_KS = np.vstack([ps_KS, PSD(plt_KS[i])])

    ps_KS_tot = ps_KS

with imageio.get_writer('./Figure/testModel_phaseDomain.gif', mode='I',fps=5) as writer:
    for r in range(MAX_STEPS):

        state = np.float32((new_observation[0::4, 0::4]).flatten())
        observation = new_observation

        action = trainer.get_action(state, Test=Test)
        new_observation = ks.advance(observation,action)

        distance = - np.linalg.norm(new_observation - target_point)

        if r % 50 == 0:
            print(r, distance)

        if Save and ((r < 100 and r % 1 == 0) or (r < 300 and r % 2 == 0) or (r > 300 and r % 5 == 0)):
            plt_KS = np.delete(plt_KS,0,0)
            plt_KS = np.vstack([plt_KS, [new_observation]])
            
            ps_KS = np.delete(ps_KS,0,0)
            ps_KS = np.vstack([ps_KS,PSD(new_observation)])
            ps_KS_tot = np.vstack([ps_KS_tot, PSD(new_observation)])

            ax1.cla()

            ax1.text(ps[0,0], ps[0,1], ps[0,2],'$E_0$',size=18, zorder=1, color='k')
            ax1.text(ps[1,0], ps[1,1], ps[1,2],'$E_g$',size=18, zorder=1, color='k')

            ax1.plot3D(ps_KS_tot[:,0], ps_KS_tot[:,1], ps_KS_tot[:,2], 'gray')
            ax1.plot3D(ps[:,0], ps[:,1], ps[:,2], 'or', markersize = 5)
            
            ax1.set_xlim(0, 2000)
            ax1.set_ylim(0, 2000)
            ax1.set_zlim(0, 2000)

            ax1.tick_params(axis='both', labelsize=12)
            ax1.set_xlabel('$\hat{e}_{(0,1)}$',fontsize=20, labelpad=10)
            ax1.set_ylabel('$\hat{e}_{(1,1)}$',fontsize=20, labelpad=10)
            ax1.set_zlabel('$\hat{e}_{(1,0)}$',fontsize=20, labelpad=10)
            
            if Plot:
                plt.pause(0.01)
                plt.show(block=False)

            plt.savefig('foo.png')
            
            image = imageio.v2.imread('foo.png')
            writer.append_data(image)