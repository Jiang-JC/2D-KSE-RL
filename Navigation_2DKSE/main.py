import numpy as np
import torch
import gc
import matplotlib.pyplot as plt

from KS2D import KS
import train
import buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Use GPU or CPU

#Select initial point and target point
init_point = np.loadtxt('./FixedPoints/1fixed_point.dat')
target_point = np.loadtxt('./FixedPoints/2fixed_point.dat')

MAX_STEPS = 1000             #total steps in one epoch
MAX_EPISODES = 1001          #total episodes
N_SAMPLE = 64                #number of sample points
S_DIM = 256                  #number of equispaced sensors (16 * 16)
A_DIM = 36                   #number of equispaced actuators (6 * 6)
A_MAX = 3                    #maximum amplitude for the actuation [-A_MAX, A_MAX]
parallel = 10                #parallel Reinforcement Learning Number

# L = Domain length
ks = KS(L = 20, N = N_SAMPLE, a_dim = A_DIM)  #2D Kuramoto-Sivashinsky class initialization

print('State Dimensions :- ', S_DIM)
print('Action Dimensions :- ', A_DIM)
print('Action Max :- ', A_MAX)

#Random Initial Seed
torch.manual_seed(42)
np.random.seed(42)

Plot = False                 #if true plot the target solution and the actual solution
Test = False                 #Test mode: load and use the policy w/o optimization

if Plot:
    x = np.arange(0., N_SAMPLE) * 20 / (N_SAMPLE - 1) 		#Grid points in x
    y = np.arange(0., N_SAMPLE) * 20 / (N_SAMPLE - 1)		#Grid points in y
    X, Y = np.meshgrid(x, y, indexing='ij')		            #Meshgrid in x-y

#parameters used to judge and record the effects of reinforcement learning
best_distance_episode = - 1e8
best_distance = - 1e8
distance_num = []
best_distance_num = []

ram = buffer.MemoryBuffer()                                         #memory class initialization
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device, Test)     #RL class initialization 

distances = 10 * [- np.linalg.norm(init_point - target_point)]

for _ep in range(MAX_EPISODES):

    #load initial condition
    new_observations = np.array([init_point] * parallel)

    #parameters used to judge and record the effects of reinforcement learning
    best_distance_step = -1e8

    for r in range(MAX_STEPS):

        states = np.float32([(new_observations[i, 0::4, 0::4]).flatten() for i in range(parallel)])
        observations = new_observations

        actions = trainer.get_action(states, Test=Test)

        new_observations = ks.advance(observations, actions)
        new_states = np.float32([(new_observations[i, 0::4, 0::4]).flatten() for i in range(parallel)])

        reward_fixs = [0] * parallel
        for parallel_pos in range(parallel):
            reward_fixs[parallel_pos] = - np.linalg.norm(new_observations[parallel_pos] - target_point)

        distances = reward_fixs

        if max(distances) > best_distance_step:

            max_pos = distances.index(max(distances))
            best_distance_step = max(distances)
            if best_distance_step > best_distance:
                best_distance = best_distance_step

        rewards = reward_fixs

        for i in range(parallel):
            trainer.ram.add(states[i], actions[i], rewards[i], new_states[i], Test)

        #Perform optimization
        trainer.optimize(Test)

        #Plot Part
        if Plot and r % 20 == 0:
            plt.clf()
            plt.contourf(X, Y, new_observations[0])
            plt.pause(0.05)
            plt.show(block=False)

    #Caluculate the steady level of the final point of this episode
    distances = [0] * parallel
    for parallel_pos in range(parallel):
        distances[parallel_pos] = - np.linalg.norm(new_observations[parallel_pos] - target_point)

    #record the effects of reinforcement learning
    distance_num.append(max(distances))
    best_distance_num.append(best_distance_step)

    if best_distance_step >= best_distance:
        best_distance = best_distance_step
        trainer.save_best_models()

    trainer.noise_size = max(min(min(- best_distance / 50, 2), trainer.noise_size), 1.2)

    if _ep % 20 == 0:
        trainer.save_models()
        np.savetxt('./Buffer/note.dat', [_ep, r, best_distance])
        np.savetxt('./Buffer/distance_num.dat', distance_num)
        np.savetxt('./Buffer/best_distance_num.dat', best_distance_num)
    
    gc.collect()                     # Recover Gabage 
    print('eposide:', _ep, 'steps:', r, 'rew:', np.float32(max(reward_fixs)), 'distance:', np.float32(max(distances)),
           'best_distance_step:', np.float32(best_distance_step), 'memory:', np.float32(trainer.ram.len/trainer.ram.maxSize*100),
           '% ', 'best_distance:', np.float32(best_distance),'noise:', np.float32(trainer.noise_size),)            

#parameters used to judge and record the effects of reinforcement learning
np.savetxt('./Buffer/distance_num.dat', distance_num)
np.savetxt('./Buffer/best_distance_num.dat', best_distance_num)

print('Completed episodes')