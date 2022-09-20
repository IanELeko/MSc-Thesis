import scipy.integrate
import numpy as np
import tensorflow as tf

class SimpleLimitCycle():
    
    def __init__(self, x1c, x2c):
        '''Centre of the limit cycle is at the provided (x1c, x2c) coordinates'''
        self.x1c = x1c
        self.x2c = x2c
    
    def ODEf(self, t, x):
        '''ODE defining the simple limit cycle
           Assumes x is a 2d vector'''
        assert(len(x)) == 2

        # defines variables x1 := (x1 - x1c); x2 := (x2 - x2c)
        x1 = x[0] - self.x1c
        x2 = x[1] - self.x2c

        # calculates the radius
        r2 = x1 ** 2  + x2 ** 2

        # gets the time derivatives
        x1dot = -x2 + x1 * (1 - r2)
        x2dot = +x1 + x2 * (1 - r2)

        return np.array([x1dot, x2dot])

    def init_solver(self, x0, t0, tN, tStep):
        '''Initialise RK45 solver for the limit cycle'''
        '''Assumes the uniform time step throughout integration'''
        self.solver = scipy.integrate.RK45(
            fun=self.ODEf,
            t0=t0,
            y0=x0,
            t_bound=tN,
            first_step=tStep,
            max_step=tStep
        )
    
    def solve_nsteps(self, x0, tStep, nSteps):
        '''get n steps of solution to the limit cycle given a time step'''
        

        # initialise the solution array
        sol = np.zeros((nSteps, 2))
        sol[0] = x0

        for i in range(1, nSteps):
            if self.solver.status == 'finished':
                print(f"ERROR: solver finished prematurely at the step {i} out of {N}")

            self.solver.step()
            sol[i] = self.solver.y
        
        return sol

    def get_nsteps(self, x0, tStep, nSteps):
        self.init_solver(x0=x0, 
                         t0=0,
                         tN=nSteps*tStep,
                         tStep=tStep)
        return self.solve_nsteps(x0, tStep, nSteps)
