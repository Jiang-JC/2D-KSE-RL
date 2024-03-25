import numpy as np


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.005, sigma = 0.1, init=1, dt=1e-1):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = np.ones(self.action_dim)*init      

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx*self.dt
        return self.X
    
    def iniOU(self,count):
        for i in range(count):
            self.X = self.sample()
        return self.X   


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1,init=0.05,theta=0.005,sigma=0.001)
    states = ou.X

    for i in range(100000):
        states = np.append(states*(np.random.randint(2,size=1)*2-1),ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
