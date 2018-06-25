""" Notes """
## TO DO:
## ...
## Maybe make a dictionary for constraints:
## - cons = {'input'       : {'weight'    : Q_du   ,
##                            'du_max'    : du_max , ... }
##           'soft-output' : {'weight_sl' : Q_sl   , 
##                            'weight_su' : Q_du   , ... }
##          }
## Could be an idea at least.
## ...
##


""" Imports """
import time, datetime

import numpy             as np
import matplotlib.pyplot as plt

from scipy.linalg   import expm
from scipy.sparse   import diags

## CVXOPT import and options
from cvxopt         import matrix, spmatrix, solvers
from cvxopt.lapack  import potrf, potri
#solvers.options['show_progress'] = False
#solvers.options['maxiters'] = 100
#solvers.options['solver'] = 'mosek'


## Discrete linearisation from continuous linearisation using matrix
## exponential. (scipy.linalg.expm)
## ...
## This should include A, B, E, G, C, Cz. Right now it misses E and G for
## simplicity.
def dicretisation( A_c, B_c, C_c, Cz_c, E_c=None, G_c=None, T_s=0.1 ):
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
    Phi = matrix( expm( M*T_s ) )

    ## Output definitions
    A  = Phi[:na,:ma]
    B  = Phi[:na,ma:ma+mb]
    E  = Phi[:na,ma+mb:ma+mb+me]
    G  = G_c
    C  = C_c
    Cz = Cz_c

    return A, B, E, G, C, Cz

## Solves Discrete Algebraic Ricatti Equation (DARE)
## P = APA' + Q - (APC' + S)(CPC' + R)^-1(APC' + S)'
## ...
## This particular example is an iterative solver:
## While P_kp1-P_k > eps
##     P_kp1 = AP_kA' + Q - (AP_kC' + S)(CP_kC' + R)^-1(AP_kC' + S)'
##     P_k = P_kp1
## ...
def dare( A, B, C, Q, R, S, maxiter=1e+5 ):
    iter    = 0
    nx, mx = A.size
    P      = matrix( spmatrix( 1.0, range(nx), range(nx) ) )
    P_km1  = matrix( 0.0, (nx,nx) )
    while np.max(np.abs(P-P_km1)) > 1e-8 and iter<maxiter:
        P_km1 = P
        term_1 = A*P_km1*A.T + Q
        term_2 = A*P_km1*C.T + S
        term_3 = C*P_km1*C.T + R
        P = term_1 - term_2*matrix(np.linalg.inv(term_3))*term_2.T
        iter += 1
    if iter >= maxiter:
        print( 'Exitted dare without converging! Reached maximum iterations of %d' % maxiter )
    else:
        print( 'dare converged to tolerance of 1e-8. Used %d iterations.' % iter )
    return matrix(P)

## Kalman filtering method for the controller
def kalman_filter( y_k, x_km1_km1, w_km1_km1, u_km1, \
                    P, A, B, G, C, Q, R, S ):
    ## Init
    y_k       = matrix(y_k)
    x_km1_km1 = matrix(x_km1_km1)
    w_km1_km1 = matrix(w_km1_km1)
    u_km1     = matrix(u_km1)

    ## Prediction and correction step
    #P = dare( A, B, C, Q, R, S )
    R_e_k = C*P*C.T + R
    R_e_k_inv = +R_e_k
    potri( R_e_k_inv )
    #R_e_k_inv = matrix(np.linalg.inv(R_e_k))
    K_fx_k = P*C.T*R_e_k_inv
    K_fw_k = S*R_e_k_inv

    x_k_km1 = A*x_km1_km1 + B*u_km1 + G*w_km1_km1
    y_k_km1 = C*x_k_km1

    e_k = y_k - y_k_km1
    x_k_k = x_k_km1 + K_fx_k*e_k
    w_k_k = K_fw_k*e_k

    return x_k_k, w_k_k

## Computation of phi_x and phi_w
def phi_function( n=30, A=[0], B=[0], G=[0], Cz=[0] ):
    ## Pre-computations
    A  = matrix(A)
    B  = matrix(B)
    G  = matrix(G)
    Cz = matrix(Cz)

    ## Initialisation and pre-computations and -allocations
    CzA    = Cz*A
    CzB    = Cz*B
    CzG    = Cz*G

    r1, c1 = CzA.size
    r2, c2 = CzG.size
    r3, c3 = CzB.size

    ## Pre-allocation
    Phi_x  = matrix( 0.0, (n*r1, c1  ) )
    Phi_w  = matrix( 0.0, (n*r2, c2  ) )
    Gamma  = matrix( 0.0, (n*r3, n*c3) )

    ## Phi_x
    Phi_x[:r1,:] = CzA
    for i in range(1,n):
        Phi_x[i*r1:(i+1)*r1,:] = Phi_x[(i-1)*r1:i*r1,:]*A

    ## Phi_w
    Phi_w[:r2,:] = CzG
    for i in range(1,n):
        Phi_w[i*r2:(i+1)*r2,:] = Phi_x[(i-1)*r2:i*r2,:]*G

    ## Gamma
    Gamma[:r3,:c3] = CzB
    for i in range(1,n):
        Gamma[i*r3:(i+1)*r3,:c3] = Phi_x[i*r1:(i+1)*r1,:]*B
    for i in range(1,n):
        Gamma[i*r3:,i*c3:(i+1)*c3] = Gamma[:(n-i)*r3,:c3]

    return Phi_x, Phi_w, Gamma

## Basic unconstrained mpc setup
def mpc_setup( x_k_k=None, w_k_k=None, u_km1=None, Phi_x=None, Phi_w=None, Gamma=None, r=None, U_min=None, U_max=None, du_max=None, soft_bounds=None, n=None, Q_z=None, Q_du=None, Q_sl=None, Q_su=None, Q_tl=None, Q_tu=None, constraints=None ):
    ## Init
    M     = 1e+8
    I_n   = matrix( spmatrix( 1.0, range(n), range(n) ) )
    Z_bar = matrix( np.kron(  matrix( 1.0, (n,1) ), r ) )

    ## Z_k definition
    b_k = matrix( Phi_x*x_k_k + Phi_w*w_k_k )
    c_k = Z_bar - b_k;

    ## Phi_z
    W_z     = +Q_z
    potrf( W_z ) # Cholesky factorisation
    W_z_bar = matrix( np.kron(I_n,W_z) )

    H_z     = matrix( (W_z_bar*Gamma).T*(W_z_bar*Gamma) )
    g_z     = matrix( -(W_z_bar*Gamma).T*(W_z_bar*c_k)  )

    ## Setup and assemble QP (Both objective function and possible constraints)
    ## ...
    ## General unconstrained MPC setup
    if (constraints == None or \
       constraints == 'input' or \
       constraints == 'soft_output' or \
       constraints == 'all'):
        ## Constraints
        tmp = matrix( spmatrix( 1.0, range(u_km1.size[0]*n), range(u_km1.size[0]*n) ) )
        A   = matrix( [tmp,-tmp] )

        LB  = matrix( np.kron( matrix( 1.0, (n,1) ), U_min ) )
        UB  = matrix( np.kron( matrix( 1.0, (n,1) ), U_max ) )
        b   = matrix( [UB,-LB] )

        ## Objective function
        H = H_z
        g = g_z

    ## Rate-of-movement (ROM) constraints
    if (constraints == 'input' or \
        constraints == 'soft_output' or \
        constraints == 'all'):
        ## Phi_du
        W_du    = +Q_du
        potrf( W_du ) # Cholesky factorisation
        W_du_bar = matrix( np.kron(I_n,W_du) )

        e      = np.ones((n,))
        Lambda = matrix( np.kron( diags([-e[:-1],e],[-1,0]).toarray(), \
                                np.eye(u_km1.size[0]) ) )

        I_0      = matrix( np.kron( I_n[:,0], np.eye(u_km1.size[0]) ) )

        ## Objective fuction
        H_du     = matrix(  (W_du_bar*Lambda).T*(W_du_bar*Lambda)    )
        g_du     = matrix( -(W_du_bar*Lambda).T*(W_du_bar*I_0*u_km1) )

        ## Hard input ROM-Constraints
        A      = matrix( [A,Lambda,-Lambda] )

        b_du_l = matrix( matrix( -du_max, (u_km1.size[0]*n,1) ) + I_0*u_km1 )
        b_du_u = matrix( matrix(  du_max, (u_km1.size[0]*n,1) ) + I_0*u_km1 )
        b      = matrix( [b, b_du_u, -b_du_l] )

        ## Objective function
        H += H_du
        g += g_du

    ## Soft output constraints
    if (constraints == 'soft_output' or \
        constraints == 'all'):
        ## Phi_s
        R_min = matrix( Z_bar - matrix( np.kron( matrix( 1.0, (n,1) ), soft_bounds ) ) )

        W_sl     = +Q_sl
        W_su     = +Q_su
        potrf( W_sl ) # Cholesky factorisation
        potrf( W_su ) # Cholesky factorisation
        W_sl_bar = matrix( np.kron( I_n, W_sl ) )
        W_su_bar = matrix( np.kron( I_n, W_su ) )

        H_s = matrix( W_su_bar.T*W_su_bar )
        g_s = matrix( W_sl_bar*matrix( 1.0, (W_sl_bar.size[0],1) ) )

        b_s_l = matrix( R_min - b_k )
        b_s_u = matrix( 1.0, (R_min.size[0],1) )*M

        ## Phi_t
        R_max = matrix( Z_bar + matrix( np.kron( matrix( 1.0, (n,1) ), soft_bounds ) ) )

        W_tl     = +Q_tl
        W_tu     = +Q_tu
        potrf( W_tl ) # Cholesky factorisation
        potrf( W_tu ) # Cholesky factorisation
        W_tl_bar = matrix( np.kron( I_n, W_tl )       )
        W_tu_bar = matrix( np.kron( I_n, W_tu )       )

        H_t = matrix( W_tu_bar.T*W_tu_bar )
        g_t = matrix( W_tl_bar*matrix( 1.0, (W_tl_bar.size[0],1) ) )

        b_t_l = matrix( -1.0, (R_min.size[0],1) )*M
        b_t_u = matrix( R_min - b_k )

        ## Constraints
        nG, mG = Gamma.size
        nL, mL = Lambda.size
        O_nGmG = matrix( 0.0, (nG,mG) )                          # Zeros matrix of same size as Gamma
        I_nGmG = matrix( spmatrix( 1.0, range(nG), range(nG) ) ) # Identity of same size as Gamma
        O_nLmL = matrix( 0.0, (nL,mL) )                          # Zeros matrix of same size as Lambda
        I_nLmL = matrix( spmatrix( 1.0, range(nL), range(nL) ) ) # Identity of same size as Lambda

        A_ = matrix( [ [ I_nLmL, Lambda, Gamma,   Gamma  ], \
                       [ O_nLmL, O_nGmG, I_nGmG,  O_nGmG ], \
                       [ O_nLmL, O_nGmG, O_nGmG, -I_nGmG ] ] )
        A  = matrix( [ A_, -A_ ] )

        bu_ = matrix( [ UB, b_du_u, b_s_u, b_t_u ] )
        bl_ = matrix( [ LB, b_du_l, b_s_l, b_t_l ] )
        b   = matrix( [ bu_, -bl_ ] )

        ## Objective function
        nH, mH = H.size
        O_nHmH = matrix( 0.0, (nH,mH) )
        H = matrix( [[H,O_nHmH,O_nHmH],[O_nHmH,H_s,O_nHmH],[O_nHmH,O_nHmH,H_t]] )

        g = matrix( [g,g_s,g_t] )


    if not (constraints == None or \
            constraints == 'input' or \
            constraints == 'soft_output' or \
            constraints == 'all'):
        print('You did not select any valid constraints...')
        print('Try again.')
        return False

    ## Return statement
    return H, g, A, b, b_k

## Simple solver to the unconstrained MPC
def qp_solver( H, g, A, b ):
    ## Compute solution to qp
    U = solvers.qp( H, g, A, b )['x']
    return U

def mpc_compute( y_k=None, x_k_k=None, w_k_k=None, u_k=None, Phi_x=None, Phi_w=None, Gamma=None, \
                 U_min=None, U_max=None, du_max=None, soft_output=None, r=None, n=None, \
                 Q_z=None, Q_du=None, Q_sl=None, Q_su=None, Q_tl=None, Q_tu=None, \
                 P=None, A=None, B=None, G=None, C=None, Q=None, R=None, S=None, constraints=None ):
    ## Init
    nu, mu = u_k.size
    nz, mz = r.size

    ## Kalman update
    x_k_k, w_k_k = kalman_filter( y_k, x_k_k, w_k_k, u_k, \
                                  P, A, B, G, C, Q, R, S )

    ## Setup QP for MPC
    H, g, A, b, b_k = mpc_setup( x_k_k, w_k_k, u_k, Phi_x, Phi_w, Gamma, r, \
                                 U_min, U_max, du_max, soft_output, n, \
                                 Q_z, Q_du, Q_sl, Q_su, Q_tl, Q_tu, constraints )

    ## Compute optimal control
    U = qp_solver( H, g, A, b )

    ## Pre-compute outputs
    Z = Gamma * U[:(u_k.size[0]*n)] + b_k
    z = Z[:nz]
    u = U[:nu]

    ## Return
    return u, U, z, Z, x_k_k, w_k_k
