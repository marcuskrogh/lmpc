""" Imports """
import time, datetime

import numpy             as np
import matplotlib.pyplot as plt

from scipy.linalg   import expm
from scipy.sparse   import diags

## CVXOPT import and options
from cvxopt         import matrix, spmatrix, solvers
from cvxopt.lapack  import potrf, potri


""" Main MPC class """
class linear_mpc:
    def __init__( self, x_s, y_s, z_s, A_c, B_c, E_c, G_c, C_c, Cz_c, T_s=0.1, off_set=True ):
        ## Store passed values
        self.x_s  = x_s
        self.y_s  = y_s
        self.z_S  = z_s
        self.A_c  = A_c
        self.B_c  = B_c
        self.E_c  = E_c
        self.G_c  = G_c
        self.C_c  = C_c
        self.Cz_c = Cz_c

        ## Size definitions
        self.nx, _ = self.x_s.size
        self.ny, _ = self.y_s.size
        self.nz, _ = self.z_s.size
        self.nu, _ = self.B_c.size[1]

        ## Dicretisation of continuous linear state space model
        ## Defines the standard system as well
        discretisation( A_c, B_c, E_c, G_c, C_c, Cz_c, T_s=0.1 )

        ## Define off_set free system
        off_set()

        if off_set:
            self.A  = self.A_aug
            self.B  = self.B_aug
            self.E  = None
            self.G  = self.G_aug
            self.C  = self.C_aug
            self.Cz = self.Cz_aug
        else:
            self.A  = self.A_d
            self.B  = self.B_d
            self.E  = None
            self.G  = self.G_d
            self.C  = self.C_d
            self.Cz = self.Cz_d

    ## Discrete linearisation from continuous linearisation using matrix
    ## exponential. (scipy.linalg.expm)
    ## ...
    ## This should include A, B, E, G, C, Cz. Right now it misses E and G for
    ## simplicity.
    def discretisation( self, A_c, B_c, C_c, Cz_c, E_c=None, G_c=None, T_s=0.1 ):
        na, ma   = A_c.size
        nb, mb   = B_c.size

        if E_c == None:
            E_c = matrix(0.0,(na,1))
        ne, me   = E_c.size

        if G_c == None:
            G_c = matrix(0.0,(na,ma))
        ng, mg   = G_c.size

        nc, mc   = C_c.size
        ncz, mcz = Cz_c.size

        ## expm only works for numpy arrays
        M   = np.hstack( [ A_c, B_c, E_c ] )
        M   = np.vstack( [ M, np.zeros((mb+me,ma+mb+me)) ] )
        Phi = expm( M*T_s )

        ## Output definitions
        self.A_d  = matrix( Phi[:na,:ma] )
        self.B_d  = matrix( Phi[:na,ma:ma+mb] )
        self.E_d  = matrix( Phi[:na,ma+mb:ma+mb+me] )
        self.G_d  = matrix( G_c  )
        self.C_d  = matrix( C_c  )
        self.Cz_d = matrix( Cz_c )

    ## Offset free control system
    def off_set( self ):
        # Define useful matrices
        B_     = matrix( spmatrix( 1.0, range(self.nx), range(self.nx) ) )
        C_     = matrix( 0.0, ( self.ny, self.nx ) )
        Cz_    = matrix( 0.0, ( self.nz, self.nx ) )
        I_nx   = matrix( spmatrix( 1.0, range(self.nx), range(self.nx) ) )
        O_nxnx = matrix( 0.0, ( self.nx, self.nx ) )
        O_nunx = matrix( 0.0, ( self.nx, self.nu ) )
        O_nwnw = matrix( 0.0, ( self.nx, self.nx ) )

        # Define augmented A
        self.A_aug  = matrix( [ [ self.A_d, O_nxnx ], [ B_, I_nx ] ] )


        # Define augmented B
        self.B_aug  = matrix( [ self.B_d, O_nunx ] )

        # Define augmented G
        self.G_aug  = matrix( [ [ self.G_d, O_nwnw.T ], [ O_nwnw, I_nx ] ] )

        # *OPTIONAL* - Define C_aug (I don't think this is needed)
        self.C_aug  = matrix( [ [ self.C_d  ], [ C_ ] ] )
        self.Cz_aug = matrix( [ [ self.Cz_d ], [ self.Cz_ ] ] )

        ## Define deviation variables
        d = matrix( 0.0, (nx,mx) )   # Start with offset of 0
        self.X_aug = matrix( [ x - x_s, d ] ) # Define new state deviation

        ## Definition of state covariance matrix
        self.Q_aug_11 = matrix( spmatrix( 1.0, range(self.nx), range(self.nx) ))
        self.Q_aug_22 = self.Q_d
        self.Q_aug_21 = matrix( 0.0, ( self.nx, self.nx ) )
        self.Q_aug    = matrix( [ [ self.Q_aug_11,   self.Q_aug_21 ], \
                                  [ self.Q_aug_21.T, self.Q_aug_22 ] ] )
