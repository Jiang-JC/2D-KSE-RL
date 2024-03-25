import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

import train
import buffer

from PSD import PSD
from KS2D import KS
from NewtonMethod import Newton

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #use GPU or CPU

MAX_STEPS = 1000             #total steps in one epoch
MAX_EPISODES = 1000          #total episodes
N_SAMPLE = 64                #number of sample points
S_DIM = 256                  #number of equispaced sensors (16 * 16)
F_DIM = 256                  #number of features
A_DIM = 36                   #number of equispaced actuators (6 * 6)
A_MAX = 3                    #maximum amplitude for the actuation [-A_MAX, A_MAX]
parallel = 10                #distributed Reinforcement Learning Number

#L = Domain length * 2
ks = KS(L = 20, N = N_SAMPLE, a_dim = A_DIM)  #Kuramoto-Sivashinsky class initialization
JFNK_improver = Newton(L = 20, N = N_SAMPLE)

#select init solution and target solution
init_point = ks.init_point()
np.savetxt('./Buffer/init_point.dat', init_point)

print('State Dimensions :- ', S_DIM)
print('Action Dimensions :- ', A_DIM)
print('Action Max :- ', A_MAX)

#Random Initial Seed
torch.manual_seed(42)
np.random.seed(42)

Plot = False                 #if true plot the target solution and the actual solution
Test = False                 #Test mode: load and use the policy w/o optimization

if Plot:
    x = np.arange(0., 64)       				# Grid points in x
    y = np.arange(0., 64)						# Grid points in y
    X, Y = np.meshgrid(x, y, indexing='ij')					# Meshgrid in x-y


#parameters used to judge and record the effects of reinforcement learning
best_distance = - 1e8
distance_num = []
best_distance_num = []

ram = buffer.MemoryBuffer()                                                #memory class initialization
trainer = train.Trainer(S_DIM, F_DIM, A_DIM, A_MAX, ram, device, Test)     #RL class initialization 

distances = [-1000] * parallel

# Load Current Points
try:
    FixedPoints_Fourier_num = list(np.atleast_2d(np.loadtxt('./FixedPoints/FixedPoints_Fourier_num.dat')))
except:
    FixedPoints_Fourier_num = []

FixedPoints_num = []
for FixedPoint_pos in range(len(FixedPoints_Fourier_num)):
    FixedPoint = np.loadtxt('./FixedPoints/FixedPoint' + str(FixedPoint_pos) + '.dat')
    FixedPoints_num.append(FixedPoint)


for _ep in range(MAX_EPISODES):

    new_observations = np.array([init_point] * parallel)    #load initial condition

    #parameters used to judge and record the effects of reinforcement learning
    best_distance_step = -1e8

    for r in range(MAX_STEPS):

        states = np.float32([(new_observations[i, 0::4, 0::4]).flatten() for i in range(parallel)])
        observations = new_observations

        actions = trainer.get_action(states, Test=Test)

        new_observations = ks.advance(observations, actions)
        new_states = np.float32([(new_observations[i, 0::4, 0::4]).flatten() for i in range(parallel)])

        reward_fixs = [0] * parallel

        future_observations = ks.advance_no_input(new_observations)
        for parallel_pos in range(parallel):
            reward_fixs[parallel_pos] -= np.linalg.norm(np.fft.fft2(new_observations[parallel_pos]) 
                                                        - np.fft.fft2(future_observations[parallel_pos]))

        distances = reward_fixs.copy()

        if max(distances) > best_distance_step:
            max_pos = distances.index(max(distances))

            max_point = new_observations[max_pos]

            best_distance_step = max(distances)
            if best_distance_step > best_distance:
                best_distance = best_distance_step

        rewards = reward_fixs

        for i in range(parallel):
            trainer.ram.add(states[i], actions[i], rewards[i], new_states[i], Test)

        # perform optimization
        trainer.optimize(Test)

        # Plot Part
        if Plot and r % 5 == 0:
            plt.clf()
            plt.contourf(X, Y, new_observations[0])
            plt.pause(0.05)
            plt.show(block=False)

    if best_distance_step > - 45:

        # JFNK part
        JFNK_score, JFNK_point  = JFNK_improver.run(max_point)
        np.savetxt('./TestPoints/' + str(_ep) + 'RL_point.dat', max_point)

        if JFNK_score > -1e-6:

            np.savetxt('./TestPoints/' + str(_ep) + 'Converge_point.dat', JFNK_point)
            repeat_flag = False
            for invariant_pos in range(len(FixedPoints_Fourier_num)):

                modify_point1 = FixedPoints_num[invariant_pos]
                modify_point2 = np.rot90(modify_point1)
                modify_point3 = np.flipud(modify_point1)
                modify_point4 = np.flipud(modify_point2)

                for modify_point in [modify_point1, modify_point2, modify_point3, modify_point4]:
                    if np.linalg.norm(np.abs(np.fft.fft2(JFNK_point)) - np.abs(np.fft.fft2(modify_point))) < 1:

                        repeat_flag = True
                        break

                if repeat_flag:
                    break

            if not repeat_flag:
                np.savetxt('./FixedPoints/FixedPoint'+ str(len(FixedPoints_Fourier_num)) +'.dat', JFNK_point)
                FixedPoints_Fourier_num.append(PSD(JFNK_point))
                np.savetxt('./FixedPoints/FixedPoints_Fourier_num.dat', FixedPoints_Fourier_num)
                FixedPoints_num.append(JFNK_point)

    # Caluculate the steady level of the final point of this episode
    distances = [0] * parallel

    future_observations = ks.advance_no_input(new_observations)
    for parallel_pos in range(parallel):
        distances[parallel_pos] -= np.linalg.norm(np.fft.fft2(new_observations[parallel_pos]) 
                                                    - np.fft.fft2(future_observations[parallel_pos]))

    #record the effects of reinforcement learning
    distance_num.append(max(distances))
    best_distance_num.append(best_distance_step)

    if best_distance_step >= best_distance:
        best_distance = best_distance_step
        trainer.save_best_models()
        
    trainer.noise_size = max(min(min(- best_distance / 20, 2), trainer.noise_size), 1.5)

    if _ep % 20 == 0:
        # trainer.save_models()
        np.savetxt('./Buffer/info.dat', [_ep, r, best_distance])
        np.savetxt('./Buffer/distance_num.dat', distance_num)
        np.savetxt('./Buffer/best_distance_num.dat', best_distance_num)
    
    gc.collect() # Recover Gabage 
    print('eposide: ', _ep, 'steps: ', r, 'rew: ', np.float32(max(reward_fixs)), 'distance: ', np.float32(max(distances)), 'best_distance_step:',
            np.float32(best_distance_step), 'memory:', np.float32(trainer.ram.len/trainer.ram.maxSize*100),'% ', 'best_distance: ', 
            np.float32(best_distance),'noise: ', np.float32(trainer.noise_size),)            

#parameters used to judge and record the effects of reinforcement learning
np.savetxt('./Buffer/distance_num.dat', distance_num)
np.savetxt('./Buffer/best_distance_num.dat', best_distance_num)

print('Completed episodes')