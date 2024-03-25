from KS2D import KS
import numpy as np
from numpy.linalg import norm, pinv

class Newton():
    def __init__(self, L = 20, N = 64):
        self.new_x = None      # Current best x
        self.epsJ = None       # epsilon used in Jacobian approximation
        self.ndts = None       # Number of timesteps taken in period T
        self.fixT = None       # Fix T for equilibrium, rather than PO solution
        self.new_fx = None     
        self.new_tol = None    
        self.new_del = None    # current size of trust region
        self.new_nits = None   # number of newton iterations taken
        self.new_gits = None   # number of GMRES iterations taken
        self.beta_ = None
        self.j_ = None
        self.h = None
        self.v = None

        self.L = L
        self.N = N

        # KSE parameters
        self.ks = KS(L, N)

    def quick_run(self, fix_state):
        n = self.N * self.N + 1     # Dimension of system, including unknown params
        mgmres = 10                 # max GMRES iterations
        nits = 1                    # max Newton iterations
        rel_err = 1e-6              # Relative error |F|/|x|

        del_ = -1.0                 # These rarely need changing for any problem
        mndl = 1e-20
        mxdl = 1e+20
        gtol = 1e-3
        self.epsJ = 1e-6

        # Initial guesses for PO, initial T,X, num timesteps
        self.new_x = fix_state.reshape(self.N * self.N)
        self.new_x = np.insert(self.new_x, 0, 5)

        self.ndts = 3
        self.fixT = False

        self.res_num = []
        
        # scale parameters by |x| then call black box
        d = np.sqrt(self.dotprd(-1, self.new_x, self.new_x))

        tol = rel_err * d
        del_ = del_ * d
        mndl = mndl * d
        mxdl = mxdl * d

        info = 1
        info = self.NewtonHook(mgmres, n, gtol, tol, del_, mndl, mxdl, nits, info)
        return - self.new_tol, self.new_x[1:].reshape(self.N, self.N)

    def run(self, fix_state):
        n = self.N * self.N + 1  # Dimension of system, including unknown params
        mgmres = 100              # max GMRES iterations
        nits = 100                # max Newton iterations
        rel_err = 1e-12           # Relative error |F|/|x|

        del_ = -1.0  # These rarely need changing for any problem
        mndl = 1e-20
        mxdl = 1e+20
        gtol = 1e-3
        self.epsJ = 1e-6

        # Initial guesses for PO, initial T,X, num timesteps
        self.new_x = fix_state.reshape(self.N * self.N)
        self.new_x = np.insert(self.new_x, 0, 5)

        self.ndts = 20
        self.fixT = False

        self.res_num = []

        # scale parameters by |x| then call black box
        d = np.sqrt(self.dotprd(-1, self.new_x, self.new_x))

        tol = rel_err * d
        del_ = del_ * d
        mndl = mndl * d
        mxdl = mxdl * d

        info = 1
        info = self.NewtonHook(mgmres, n, gtol, tol, del_, mndl, mxdl, nits, info)

        print("All Newton Improver Finish", "  Value = ", self.new_tol)
        return - self.new_tol, self.new_x[1:].reshape(self.N, self.N)

    def dotprd(self, n_, a, b):
        n1 = 0
        if n_ == -1:
            n1 = 1
        d = np.sum(a[n1:] * b[n1:])
        return d

    def multJp(self, n, x):
        y = x.copy()
        return y

    def getrhs(self, n_, x):
        y_ = self.steporbit(self.ndts, x)
        y = y_ - x
        y[0] = 0.0
        return y

    def multJ(self, n_, dx):
        eps = np.sqrt(self.dotprd(1, dx, dx))
        eps = self.epsJ * np.sqrt(self.dotprd(1, self.new_x, self.new_x)) / eps
        y = self.new_x + eps * dx
        s = self.getrhs(n_, y)
        y = (s - self.new_fx) / eps
        
        if self.fixT:
            y[0] = 0.0
        else:
            s = self.steporbit(1, self.new_x)
            dt = self.new_x[0] / self.ndts
            s = (s - self.new_x) / dt
            y[0] = self.dotprd(-1, s, dx)
            
        return y


    def saveorbit(self):
        print(f"newton: iteration {self.new_nits}")
        
        norm_x = np.sqrt(self.dotprd(-1, self.new_x, self.new_x))
        relative_err = self.new_tol / norm_x

        # SAVE current solution, new_xd

    def NewtonHook(self, m, n, gtol, tol, del_, mndl, mxdl, nits, info):
        self.new_nits = 0 
        self.new_gits = 0 
        self.new_del  = del_ 
        mxdl_    = mxdl 
        ginfo    = info 
        self.new_fx   = self.getrhs(n,self.new_x) 
        self.new_tol  = np.sqrt(self.dotprd(n,self.new_fx,self.new_fx)) 
        
        if del_<0:  
            self.new_del = self.new_tol / 10
            mxdl_   = 1e99
        if info==1:
            print(f'newton: nits={self.new_nits}  res={self.new_tol}')
            pass

        self.saveorbit()
        x_   = self.new_x 
        fx_  = self.new_fx 
        tol_ = self.new_tol 
        tol__ = 1e99 

        if self.new_tol < tol:
            if info==1:
                # print('newton: input already converged') 
                pass
            info = 0 
            return info
                        # - - - - Start main loop - - - - -  -
        
        while True:

            if self.new_del < mndl:
                if info==1:
                    # print('newton: trust region too small') 
                    pass
                info = 3 
                return info
                        # find hookstep s and update x
            s        = np.zeros(n) 
            gres     = gtol * self.new_tol 
            gdel     = self.new_del 
            if ginfo != 2:
                self.new_gits = m 
            if del_ == 0:
                self.new_gits = 9999 
            s, gres, gdel, self.new_gits, ginfo = self.GMRESm(m, n, s, fx_, gres, gdel, self.new_gits, ginfo) 
            
            # - - - Quit setting - - -
            self.res_num.append(self.new_tol)
            if len(self.res_num) > 20 and (self.res_num[-20] - self.new_tol) / self.new_tol < 1e-3:
                print("Change too small! ", (self.res_num[-20] - self.new_tol) / self.new_tol)
                return info

            ginfo = info 
            self.new_x = x_ - s 
                        # calc new norm, compare with prediction

            self.new_fx  = self.getrhs(n, self.new_x) 
            self.new_tol = np.sqrt(self.dotprd(n, self.new_fx, self.new_fx)) 


            snrm = np.sqrt(self.dotprd(n, s, s)) 
            ared = tol_ - self.new_tol 
            pred = tol_ - gdel 
            
            if info==1:
                print(f'newton: nits={self.new_nits}  res={self.new_tol}') 
                # print(f'newton: gits={self.new_gits}  del={self.new_del}') 
                # print(f'newton: |s|={snrm}  pred={pred}') 
                # print(f'newton: ared/pred={ared/pred}') 
                pass

            if del_ == 0:
                if info==1:
                    # print('newton: took full newton step') 
                    pass
            elif self.new_tol > tol__:
                if info==1:
                    # print('newton: accepting previous step') 
                    pass
                self.new_x   = x__ 
                self.new_fx  = fx__ 
                self.new_tol = tol__ 
                self.new_del = del__ 
            elif ared < 0:
                if info==1:
                    # print("newton: reached max its\n")
                    pass
                self.new_del = snrm * 0.5
                ginfo = 2
            elif ared/pred < 0.75:
                if info == 1:
                    # print('newton: step ok, trying smaller step')
                    pass
                x__ = self.new_x.copy()
                fx__ = self.new_fx
                tol__ = self.new_tol
                if ared/pred > 0.1:
                    del__ = snrm
                if ared/pred <= 0.1:
                    del__ = snrm * 0.5
                self.new_del = snrm * 0.7
                ginfo = 2
            elif snrm < self.new_del * 0.9:
                if info == 1:
                    # print('newton: step good, took full newton step')
                    pass
                self.new_del = min(mxdl_, snrm * 2)
            elif self.new_del < mxdl_ * 0.9:
                if info == 1:
                    # print('newton: step good, trying larger step')
                    pass
                x__ = self.new_x.copy()
                fx__ = self.new_fx
                tol__ = self.new_tol
                del__ = self.new_del
                self.new_del = min(mxdl_, snrm * 2)
                ginfo = 2
            
            if ginfo == 2:
                continue
            self.new_nits += 1
            self.saveorbit()
            x_ = self.new_x.copy()
            fx_ = self.new_fx
            tol_ = self.new_tol
            tol__ = 1e99
            if self.new_tol < tol:
                if info == 1:
                    # print('newton: converged')
                    pass
                info = 0
                return info
            elif self.new_nits == nits:
                if info == 1:
                    # print('newton: reached max its')
                    pass
                info = 2
                return info
        return info


    def GMREShook(self, j, h, m, beta, del_):
        a = h[:j+2, :j+1]
        u, s, v = np.linalg.svd(a)

        p = beta * u[0, 0:j+1].T
        mu = max(s[j]*s[j]*1e-6, 1e-99)
        qn = 1e99
        
        # print(mu, s, qn, p)
        while qn > del_:
            mu = mu * 1.1

            q = p*s / (mu+s*s)
            qn = np.sqrt(np.sum(q*q))
        y = (v.T) @ q

        p = - h[0:j+2, 0:j+1] @ y
        p[0] = p[0] + beta
        del_ = np.sqrt(np.sum(p*p))

        return y, del_

    def GMRESm(self, m, n, x, b, res, del_, its, info):
        if info == 2:
            y, del_ = self.GMREShook(self.j_, self.h, m, self.beta_, del_)
            z = self.v[:, :self.j_] @ y[:self.j_]
            x = self.multJp(n, z)
            info = 0

            return x, res, del_, its, info
        tol = res
        imx = its
        its = 0
        self.v = np.zeros((n, m + 1))
        while True:  # restart
            res_ = 1e99
            stgn = 1.0 - 1e-14
            self.beta_ = np.sqrt(self.dotprd(n, x, x))
            if self.beta_ == 0.0:
                w = 0.0
            else:
                w = self.multJ(n, x)
            w = b - w
            self.beta_ = np.sqrt(self.dotprd(n, w, w))
            self.v[:, 0] = w / self.beta_
            self.h = np.zeros((m + 1, m))
            for j in range(m):
                self.j_ = j
                its += 1
                z = self.v[:, j:j+1]
                z = self.multJp(n, z)

                z = z.reshape(n)
                w = self.multJ(n, z)

                for i in range(j + 1):
                    self.h[i, j] = self.dotprd(n, w, self.v[:, i])
                    w -= self.h[i, j] * self.v[:, i:i+1].reshape(n)

                self.h[j+1, j] = np.sqrt(self.dotprd(n, w, w))
                self.v[:, j+1] = w / self.h[j+1, j]

                p = np.zeros((j + 2, 1))
                p[0] = self.beta_

                h_ = self.h[:j+2, :j+1]
                y = pinv(h_) @ p

                p = -self.h[:j+2, :j+1] @ y
                p[0] += self.beta_
                res = np.sqrt(np.sum(p ** 2))

                if info == 1 and its % 10 == 0:
                    # print(f'gmresm: it={its}  res={res}')
                    pass
                done = (res <= tol) or (its == imx) or (res > res_)
                if done or j == m - 1:
                    if del_ > 0.0:
                        y, del_ = self.GMREShook(j, self.h, m, self.beta_, del_)
                    
                    z = self.v[:, :j+1].dot(y[:j+1])
                    z = self.multJp(n, z)
                    x += z

                    if its == imx:
                        info = 2
                    if res > res_:
                        info = 1
                    if res <= tol:
                        info = 0
                    if done:

                        return x, res, del_, its, info
                    if del_ > 0:
                        # print("gmres: WARNING: m too small. restart affects hookstep.")
                        pass
                res_ = res * stgn
        return x, res, del_, its, info

    def steporbit(self, ndts_, x_state):
        # if ndts_ !=1:
        #     dt = x_state[0] / ndts_

        u = x_state[1:].reshape(self.N, self.N)
        for n in range(ndts_):
            u =self.ks.advance_no_input(u)

        y = np.zeros(x_state.shape)
        y[1:] = u.reshape(self.N * self.N)
        
        return y