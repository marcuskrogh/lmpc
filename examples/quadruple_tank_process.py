""" Imports """
import numpy as np

from scipy.integrate import odeint
from scipy.optimize  import fsolve
from cvxopt          import matrix


""" Quadruple tank process """
class qtp:
    def __init__( self, x0=[0, 0, 0, 0] ):
        '''Set initial conditions and plant parameters.'''
        # Initial state
        self.x     = np.array(x0).reshape(4,) # [cm]
        self.t     = 0

        # Parameters
        self.gamma = np.array([0.403, 0.31])    # [%]
        self.A     = 380.1327                   # [cm^2]
        self.a     = 1.2272                     # [cm^2]
        self.g     = 981.0                      # [cm/s^2]
        self.rho   = 1.0                        # [g/cm^3]

    def state( self, x, t, u, d ):
        '''Ordinary differential equation model.'''
        # Influent from pump and disturbances
        qin = np.array(   [
            self.gamma[0]    *u[0], # Valve 1 to tank 1 [cm^3/s]
            self.gamma[1]    *u[1], # Valve 2 to tank 2 [cm^3/s]
            (1-self.gamma[1])*u[1], # Valve 2 to tank 3 [cm^3/s]
            (1-self.gamma[0])*u[0], # Valve 1 to tank 4 [cm^3/s]
                        ]   )

        # Liquid levels
        h = x/(self.rho*self.A) # [cm]
        qout = np.array(self.a*np.sqrt(2*self.g*h))

        # Influent from other tanks
        qoutin = np.array([
        qout[2],
        qout[3],
        0,
        0
        ])

        # Mass balance
        # from pump + from tanks -effluent
        dxdt = self.rho*(qin + qoutin - qout)
        return np.array(dxdt).reshape((4,))

    def measurement( self, x ):
        return x/(self.rho*self.A)

    def output( self, x ):
        return (x/(self.rho*self.A))[:2]

    def simulation_step( self, dt, u, d ):
        '''Simulate dt seconds of the ODE model.'''
        #u = np.array(u)
        #d = np.array(d)
        # Explicit solver does not use Jacobian so theres is no reason to include it
        sol     = odeint( self.state, self.x, [0, dt], args=(u,d,) ) # [initial, final]
        self.x  = sol[1] # Save final as current state
        self.t += dt
        return self.x

    def continuous_linearisation( self, t=0, u=[150,150], d=[0.0,0.0] ):
        ## Compute steady states
        u_s = u
        d_s = d
        x_s = matrix(fsolve( self.state, self.x, args=( t, u_s, d_s, ) ))
        y_s = matrix(self.measurement( x_s ))
        z_s = matrix(self.output( x_s ))

        ## Continuous linearisation
        T = matrix(self.A * np.sqrt( 2 * self.g * y_s) / ( self.a * self.g ))
        A_c = [
            [ -1/T[0],  0        ,  1/T[2],  0         ],
            [  0        , -1/T[1],  0        ,  1/T[3] ],
            [  0        ,  0        , -1/T[2],  0         ],
            [  0        ,  0        ,  0        , -1/T[3],]]
        A_c = matrix(A_c)

        B_c =   [
            [ self.rho * self.gamma[0]     , 0                                ],
            [ 0                            , self.rho * self.gamma[1]         ],
            [ 0                            , self.rho * ( 1 - self.gamma[1] ) ],
            [ self.rho * ( 1 - self.gamma[0] ) , 0                            ],
                ]
        B_c = matrix(B_c)

        C_c = np.diagflat( 1/(self.rho * self.A)*np.ones((self.x.shape[0])), 0 )
        C_c = matrix(C_c)

        Cz_c = matrix(C_c[0:z_s.size[0],:])

        return x_s, y_s, z_s, A_c.T, B_c.T, C_c, Cz_c
