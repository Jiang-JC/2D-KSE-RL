import numpy as np

class KS:
    def __init__(self, L = 20, N = 64, h = 0.05, a_dim = 64, nt = 1000):
        # Parameters
        self.rho = 1
        self.R = 1
        self.N = N
        self.L = L
        self.h = 5e-2
        self.dx = L / N
        self.x = L * np.arange(0, N)[:, np.newaxis] / (N - 1)
        self.y = L * np.arange(0, N)[:, np.newaxis] / (N - 1)
        self.X, self.Y = np.meshgrid(self.x, self.y)


        kx1 = np.mod(1 / 2 + np.arange(self.N) / self.N, 1) - 1 / 2
        ky1 = np.mod(1 / 2 + np.arange(self.N) / self.N, 1) - 1 / 2
        kx = kx1 * (2 * np.pi / self.dx)
        ky = ky1 * (2 * np.pi / self.dx)
        self.KX, self.KY = np.meshgrid(kx, ky)

        # Anti-aliasing filter
        self.AA = (np.abs(self.KX) < (2 / 3) * np.max(kx)) * (np.abs(self.KY) < (2 / 3) * np.max(ky))

        # Precompute ETDRK4 scalar quantities
        L_ = -self.rho / self.R * (-self.KX ** 2 - self.KY ** 2) - 1 / self.R * (self.KX ** 4 + 2 * (self.KX ** 2) * (self.KY ** 2) + self.KY ** 4)
        self.E = np.exp(h * L_)
        self.E2 = np.exp(h * L_ / 2)
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        rr = np.repeat(r[np.newaxis, np.newaxis, :], N, axis=0).repeat(N, axis=1)
        LL = np.repeat(L_[:, :, np.newaxis], M, axis=2)
        LR = h * LL + rr

        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=2))
        self.f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=2))
        self.f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=2))
        self.f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=2))

        # self.AEnorm = np.linalg.norm(np.real(np.fft.ifft2(Fphi)))
        # self.At = np.array([0])

        self.a_dim = a_dim
        self.B = np.zeros((self.N, self.N, a_dim))
    
        sig = 2.4

        unit_disx = L / N
        unit_disy = L / N

        gaus = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                gaus[i][j] = 1/(2*np.pi*(sig**2))*np.exp(-0.5*((unit_disx*(i-self.N//2))**2+
                    (unit_disy*(j-self.N//2))**2)/(sig**2))
                
        roll_dis = self.N//np.sqrt(a_dim)

        for i in range(0, a_dim):
            roll_x=i%np.sqrt(a_dim) - np.sqrt(a_dim)//2
            roll_y=i//np.sqrt(a_dim) - np.sqrt(a_dim)//2
            self.B[:,:,i] = gaus
            self.B[:,:,i] = np.roll(self.B[:,:,i] , int(roll_dis*roll_x+roll_dis/2) ,axis=0)
            self.B[:,:,i] = np.roll(self.B[:,:,i] , int(roll_dis*roll_y+roll_dis/2) ,axis=1)
            self.B[:,:,i] = self.B[:,:,i]/(self.B[:,:,i].max())


    def init_point(self):
        # Initial condition
        phi = 1 * (np.sin(2 * np.pi / self.L * self.X + 2 * np.pi / self.L * self.Y) + np.sin(2 * np.pi / self.L * self.X) 
                   + np.sin(2 * np.pi / self.L * self.Y))
        Fphi = np.fft.fft2(phi)

        phi += np.random.rand(self.N, self.N)
        for i in range(1000):
            phi = self.advance_no_input(phi)

        return phi

    def advance_no_input(self, phi):
        Fphi = np.fft.fft2(phi)
        Nv = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * Fphi)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * Fphi)) ** 2))
        a = self.E2 * Fphi + self.Q * Nv
        Na = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * a)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * a)) ** 2))
        b = self.E2 * Fphi + self.Q * Na
        Nb = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * b)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * b)) ** 2))
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * c)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * c)) ** 2))
        Fphi = self.E * Fphi + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        if phi.ndim == 2:
            Fphi[0, 0] = 0
        elif phi.ndim == 3:
            Fphi[:, 0, 0] = 0
        phi = np.real(np.fft.ifft2(Fphi))
        # Ephi = np.linalg.norm(np.real(np.fft.ifft2(Fphi)))
        # self.AEnorm = np.append(self.AEnorm, Ephi)
        # At = np.append(At, t)
        # print(phi, new_phi)
        return phi

    def advance(self, phi, action):

        if action.ndim == 1:
            
            self.f0 = np.zeros((self.N, self.N, 1))
            dum = np.zeros((self.N, self.N, self.a_dim))
            dum = self.B[:, :, :] * action[:]

            self.f0 = np.sum(dum, axis=2)
            self.f = np.fft.fft2(self.f0)

        else:
            # forcing shape
            self.f0 = np.zeros((len(action), self.N, self.N))
            dum = np.zeros((self.N, self.N, self.a_dim))

            for parallel_pos in range(len(action)):

                dum = self.B[:, :, :] * action[parallel_pos, :]
                self.f0[parallel_pos] = np.sum(dum, axis=2)

            self.f = np.fft.fft2(self.f0)

        Fphi = np.fft.fft2(phi)
        Nv = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * Fphi)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * Fphi)) ** 2))
        a = self.E2 * Fphi + self.Q * Nv
        Na = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * a)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * a)) ** 2))
        b = self.E2 * Fphi + self.Q * Na
        Nb = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * b)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * b)) ** 2))
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = -0.5 * (np.fft.fft2(np.real(np.fft.ifft2(1j * self.KX * self.AA * c)) ** 2) 
                     + np.fft.fft2(np.real(np.fft.ifft2(1j * self.KY * self.AA * c)) ** 2))
        Fphi = self.E * Fphi + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3 + self.h * self.f
        if phi.ndim == 2:
            Fphi[0, 0] = 0
        elif phi.ndim == 3:
            Fphi[:, 0, 0] = 0
        phi = np.real(np.fft.ifft2(Fphi))
        # Ephi = np.linalg.norm(np.real(np.fft.ifft2(Fphi)))
        # self.AEnorm = np.append(self.AEnorm, Ephi)
        # At = np.append(At, t)

        return phi
