## General imports
import numpy             as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

## MPC related functions
import sys
sys.path.append('../')
from lmpc_methods import *

## Import test model - Quadruple Tank Process (QTP)
from quadruple_tank_process import qtp

def closed_loop_simulation_tf( x0, u0, A, B, G, C, Cz, Q, R, x_true_0, A_true, B_true, G_true, C_true, Cz_true, w_k, v_k, \
                               T_s, N, n, r, r_c, disturbance, u_min, u_max, du_max, soft_bounds, constraints=None ):
    ## Initialisation: Model parameters and values
    X = x0    # Initial value:    State
    U = u0    # Initial value:    Input
    Y = C*X   # Initial value:    Measurement
    Z = Cz*X  # Initial value:    Output
     
     
    ## MPC TOOLBOX - Phi-function of augmented model
    Phi_x, Phi_w, Gamma = phi_function( A=A, B=B, G=G, Cz=Cz, n=n )    
 
    ## Re-define sizes
    nX, _ = A.size
    _, nU = B.size
    nY, _ = C.size
    nZ, _ = Cz.size
 
    ## Desired level
    r_    = matrix( [ r     ] )
    U_min = matrix( [ u_min ] )
    U_max = matrix( [ u_max ] )
 
    ## Assuming no co-variance between state and measurement
    S = matrix( 0.0, ( nU, nY ) ) # Kalman filter:    Cov(W,V)
 
    ## Generate covariance matrix P
    P = dare( A, B, G, C, Q, R, S )
 
    ## MPC TOOLBOX - Weights for objective and constraints
    Q_z   = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Weight for QP: || Z_bar - Z ||^2_Q_z
    Q_du  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Weight for input rate-of-change regularsation
    Q_sl  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Weight for lower soft output bound ( Linear    )
    Q_su  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Weight for lower soft output bound ( Quadratic )
    Q_tl  = matrix( spmatrix( 1e+5, range(nZ), range(nZ) ) ) # Weight for upper soft output bound ( Linear    )
    Q_tu  = matrix( spmatrix( 1e+5, range(nZ), range(nZ) ) ) # Weight for upper soft output bound ( Quadratic )
     
    ## Init
    dt    = T_s                        # Step for measurement (Solving state equation)
    X_k_k = X                          # Initial state
    x_k   = x_true_0
    W_k_k = matrix(  0.0, ( nU, 1 ) )  # Initial value
    U_k   = U                          # Initial value
 
    ## Pre-allocation for plotting
    U_plot  = matrix(  0.0, ( N, nU ) ) # Storage of U for plotting
    Z_plot  = matrix(  0.0, ( N, nZ ) ) # Storage of computed output
    Z2_plot = matrix(  0.0, ( N, nZ ) ) # Storage of measured output
 
    ## Closed loop simulation
    for i in range(0,N):
        ## Set-point change half way through.
        if i >= N/3:
            r_new  = r + r_c  # Desired output level
            r_ = matrix( [ r_new ] )  # Conversion to deviation
 
        ## Storage of input (U) for plotting
        U_plot[i,:] = U_k.T
 
        ## Simulation of system
        if i >= N-N/3:
            ## Disturbance in process two 3rds through
            x_k = A_true*x_k + B_true*(U_k+disturbance) + G_true*w_k[:,i]
            y_k = C_true*x_k + v_k[:,i]
            z_k = Cz_true*x_k
        else:
            ## No disturbance
            x_k = A_true*x_k + B_true*U_k + G_true*w_k[:,i]
            y_k = C_true*x_k + v_k[:,i]
            z_k = Cz_true*x_k
 
        # MPC TOOLBOX - Compute optimal input
        U_k, U_pred, z, Z_pred, X_k_k, W_k_k = mpc_compute( y_k, X_k_k, W_k_k, U_k, Phi_x, Phi_w, Gamma, \
            U_min, U_max, du_max, soft_bounds, r_, n, Q_z, Q_du, Q_sl, Q_su, Q_tl, Q_tu, \
            P, A, B, G, C, Q, R, S, constraints=constraints )
 
        ## PLOT - Store results
        Z_plot[i,:]  = matrix(z).T
        Z2_plot[i,:] = matrix(z_k).T
 
 
    ## Return statement
    return Z_plot, Z2_plot, U_plot
 
def closed_loop_plot_tf( Z_model, Z_true, U_plot, T_s, r, r_c, u_min, u_max ):
    ## Size definitions
    N, nz = Z_model.size
    _, nu = U_plot.size
     
    ## Init
    r_new = r + r_c
     
    T_plot = np.array(range(N))*T_s/60 # Initialise time axis (Minutes)
 
    ## Plot outputs: Initialisation
    ax1 = [None] * nz                   # Initialise axes
    fig1, ax1 = plt.subplots(nrows=nz)  # Initialise subplots
    if nz == 1:
        ax1 = [ax1]
    fig1.suptitle( 'Output timeline', fontsize=15 )
 
    ## Plot outputs: Z_1, ..., Z_nz
    pl = [None] * 3                     # Initialise axes
    for i in range(nz):
        label_1 = 'Linear model: $Z_%d$' % (i+1)
        label_2 = 'True process: $Z_%d$' % (i+1)
        name  = '$Z_%d$' % (i+1)
        pl[0], = ax1[i].plot(T_plot, Z_model[:,i], 'c-',  label=label_1 )
        pl[1], = ax1[i].plot(T_plot, Z_true[:,i],  'k-',  label=label_2 )
 
        T1     = matrix( T_plot[:int(np.ceil(N/3))]  )
        T2     = matrix( T_plot[int(np.floor(N/3)):] )
        len_1  = T1.size[0]  # Define lengths of set-points
        len_2  = T2.size[0]  # Define lengths of set-points
        r_full = matrix( [ r[i]*matrix(1.0,(len_1,1)), r_new[i]*matrix(1.0,(len_2,1)) ] )
        T_full = matrix( [ T1, T2 ] )
 
        pl[2], = ax1[i].plot(T_full, r_full, 'r--', label='Desired level'  )
        Z_min  = np.min( [ np.min(Z_model[:,i]), np.min(Z_true[:,i]) ] )
        Z_max  = np.max( [ np.max(Z_model[:,i]), np.max(Z_true[:,i]) ] )
        ax1[i].set_ylim( Z_min-0.05*Z_min, Z_max+0.05*Z_max )
        ax1[i].legend( handles=pl, fontsize=15 )
        ax1[i].legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15 ) # Place legend outside plot
        ax1[i].set_ylabel(name, rotation=0 )
        ax1[i].set_xlabel('Time (minutes)')
 
    ## Plot inputs: Initialisation
    ax2 = [None] * nu                   # Initialise axes
    fig2, ax2 = plt.subplots(nrows=nu)  # Initialise subplots
    if nu == 1:
        ax2 = [ax2]
    fig2.suptitle( 'Input timeline', fontsize=15 )
 
    ## Plot input: U_1
    p2 = [None] * 2                     # Initialise axes
    for i in range(nu):
        name = '$U_%d$' % (i+1)
        bound_interval = u_max[i]-u_min[i]     # Define bound interval
        p2[0], = ax2[i].step( T_plot, U_plot[:,i] , 'b', label='Input: U_1')
        p2[1], = ax2[i].plot( T_plot, u_min[i]+matrix(0.0,(N,1)), 'r--', label='Bounds' )
        _,     = ax2[i].plot( T_plot, u_max[i]+matrix(0.0,(N,1)), 'r--' )
        ax2[i].set_ylim( u_min[i]-0.1*bound_interval, u_max[i]+0.1*bound_interval )
        ax2[i].legend( handles=[ p2[0], p2[1] ], fontsize=15 )
        ax2[i].legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15 ) # Place legend outside plot
        ax2[i].set_ylabel(name, rotation=0 )
        ax2[i].set_xlabel('Time (minutes)')
 
    ## Show plots
    #plt.tight_layout()  # Ensure x-ticks aren't cut off
    plt.show()
     
    ## Return statement
    return fig1, ax1, fig2, ax2

def closed_loop_simulation( horison=120, x0=[5000.0, 5000.0, 5000.0, 5000.0], u0=[350.0, 350.0], \
                            model=qtp, T_s=30, n=40, r=[ 25, 20 ], r_c=[ 10, 10  ],              \
                            disturbance=[ 75.0, 75.0 ], u_min=[ 1e-2, 1e-2 ],                    \
                            u_max=[ 500.0, 500.0 ], du_max=50, soft_bounds=[ 1.0, 1.0 ],         \
                            off_set='off', Q_d=1e+4, constraints=None ):
    ## Initialisation and type setting: Model parameters and values
    x           = matrix(x0)          # Initial value:    State
    u           = matrix(u0)          # Initial value:    Input
    r           = matrix(r )          # Set-point(s)
    r_c         = matrix(r_c)         # Set-point change(s)
    disturbance = matrix(disturbance) # Disturbance(s)
    u_min       = matrix(u_min)       # Lower bound on input
    u_max       = matrix(u_max)       # Upper bound on input
    soft_bounds = matrix(soft_bounds) # Soft output bounds
    
    plot_info = { 'u_min' : 0.0,
                  'u_max' : 0.0,
                  'T_s'   : 0.0, }
    plot_info['u_min'] = u_min
    plot_info['u_max'] = u_max
    plot_info['T_s']   = T_s
    
    
    ## MPC TOOLBOX - Initialisation of toolbox
    #T_s         = 30                          # Discretisation step
    #n           = 40                          # Prediction step for MPC
    #r           = matrix( [ 25, 20 ] )        # Desired output level
    #r_c         = matrix( [ 10, 10  ] )       # Increase in set-point
    #disturbance = matrix( [ 100.0, 50.0 ] )   # Disturbance, 2/3rds through
    #u_min       = matrix( [ 1e-2, 1e-2 ] )    # Minimum allowed input
    #u_max       = matrix( [ 500.0, 500.0 ] )  # Maximum allowed input
    #du_max      = 20                          # Maximum allowed change input per step
    #soft_bounds = matrix( [ 1.0, 1.0 ] )      # Soft output bounds
    #off_set     = 'off'                       # No offset-free augmentation

    ## Initialisation: Model class and methods
    test_model = model( x )                 # Initialise model class
    y = test_model.measurement( x )         # Initial value:    Measurement
    z = test_model.output( x )              # Initial value:    Output

    ## Initialisation: Model linearisation
    x_s, y_s, z_s, A, B, C, Cz = test_model.continuous_linearisation( u=u, d=[] )
    u_s = u

    ## Initialisation: Size definitions
    nx, mx = x_s.size
    ny, my = y_s.size
    nz, mz = z_s.size
    nu, mu = u_s.size

    ## MPC TOOLBOX - Discretisation of continuous linearisation.
    A, B, E, G, C, Cz = discretisation( A, B, C, Cz, T_s=T_s ) # Generic 'Continuous to Discrete' method

    if off_set == 'on':
        ## MPC TOOLBOX - Augmented system for off-set free control - Generic code
        # Define useful matrices
        B_d    = matrix( spmatrix( 1.0, range(nx), range(nx) ) )
        C_d    = matrix( 0.0, ( ny, nx ) )
        Cz_d   = matrix( 0.0, ( nz, nx ) )
        I_nx   = matrix( spmatrix( 1.0, range(nx), range(nx) ) )
        O_nxnx = matrix( 0.0, ( nx, nx ) )
        O_nunx = matrix( 0.0, ( nx, nu ) )
        O_nwnw = matrix( 0.0, ( nx, nx ) )

        # Define augmented A
        A  = matrix( [ [ A, O_nxnx ], [ B_d, I_nx ] ] )
        

        # Define augmented B
        B  = matrix( [ B, O_nunx ] )

        # Define augmented G
        G  = matrix( [ [ G, O_nwnw.T ], [ O_nwnw, I_nx ] ] )

        # *OPTIONAL* - Define C_aug (I don't think this is needed)
        C  = matrix( [ [ C  ], [ C_d ] ] )
        Cz = matrix( [ [ Cz ], [ Cz_d ] ] )
        
        ## Define deviation variables
        d = matrix( 0.0, (nx,mx) )   # Start with 0.0 off-set
        X = matrix( [ x - x_s, d ] )
        
        ## Define size
        nd, md = d.size
        
        ## State covariance
        Q_11 = matrix( spmatrix( 1.0, range(nx), range(nx) ) ) # Construction of Q(1,1)
        Q_22 = matrix( spmatrix( Q_d, range(nd), range(nd) ) ) # Construction of Q(2,2)
        Q_21 = matrix( 0.0, ( nx, nd ) )                       # Construction of Q(2,1)
        Q    = matrix( [ [ Q_11, Q_21 ], [ Q_21.T, Q_22 ] ] )  # Kalman filter: Cov(W,W)
        
    elif off_set == 'off' or off_set == None:
        ## Define deviation variables
        X = matrix(x) - matrix(x_s)
        
        ## State covariance
        Q = matrix( spmatrix( 1.0, range(nx), range(nx) ) )    # Kalman filter: Cov(W,W)
        
    ## Define deviation variables
    Y = matrix(y) - matrix(y_s)
    Z = matrix(z) - matrix(z_s)
    U = matrix(u) - matrix(u_s)
    
    ## MPC TOOLBOX - Phi-function of augmented model
    Phi_x, Phi_w, Gamma = phi_function( A=A, B=B, G=G, Cz=Cz, n=n )    

    ## Re-define sizes
    nX, mX = X.size
    nU, mU = (nu, mu)
    nY, mY = (ny, my)
    nZ, mZ = (nz, mz)

    ## Desired level
    r_    = matrix( [ r     - z_s ] )
    U_min = matrix( [ u_min - u_s ] )
    U_max = matrix( [ u_max - u_s ] )

    ## Define weight matrices - Model dependent! *** THESE MAY NEED CHANGING ***
    R    = matrix( spmatrix(  1.0, range(nY), range(nY) ) )   # Kalman filter:    Cov(V,V)
    S    = matrix( 0.0, ( nX, nY ) )                          # Kalman filter:    Cov(W,V)

    ## Generate covariance matrix P
    P = dare( A, B, C=C, Q=Q, R=R, S=S )

    ## MPC TOOLBOX - Weights for objective and constraints
    Q_z   = matrix( spmatrix( 1e+2, range(nZ), range(nZ) ) ) # Z weight for QP: || Z_bar - Z ||^2_Q_z
    Q_du  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # du weight for hard input change constraint
    Q_sl  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Soft output bound constraint
    Q_su  = matrix( spmatrix( 1e+0, range(nZ), range(nZ) ) ) # Soft output bound constraint
    Q_tl  = matrix( spmatrix( 1e+3, range(nZ), range(nZ) ) ) # Soft output bound constraint
    Q_tu  = matrix( spmatrix( 1e+2, range(nZ), range(nZ) ) ) # Soft output bound constraint
    
    ## Init
    dt    = T_s                        # Step for measurement (Solving state equation)
    X_k_k = X                          # Initial state
    W_k_k = matrix(  0.0, ( nX, mX ) ) # Initial value
    U_k   = U                          # Initial value
    N     = horison                    # Amount of simulation steps

    ## Pre-allocation for plotting
    T_plot  = matrix(  0.0, ( N+1, 1) ) # Storage of time
    U_plot  = matrix(  0.0, ( N, nU ) ) # Storage of U for plotting
    Y_plot  = matrix(  0.0, ( N, nY ) ) # Storage of computed output
    R_plot  = matrix(  0.0, ( N, nZ ) ) # Storage of set-points

    ## Closed loop simulation
    y_var = 1e-2**2
    u_var = 1e-2**2
    Q_    = y_var*np.eye(nY)
    R_    = u_var*np.eye(nU)
    T_0   = time.time()
    for i in range(0,N):
        ## Compute noises
        v_k = matrix( np.random.multivariate_normal(np.zeros(nY),Q_) )*dt
        w_k = matrix( np.random.multivariate_normal(np.zeros(nU),R_) )*dt
        
        ## Set-point change
        if   i == int(np.round(N/7)):
            r_ += r_c
            r  += r_c
        elif i == int(np.round(2*N/7)):
            r_ -= r_c
            r  -= r_c
        elif i == int(np.round(3*N/7)):
            r_ -= r_c
            r  -= r_c
        elif i == int(np.round(4*N/7)):
            r_ += r_c
            r  += r_c

        ## Storage of input (U) for plotting
        U_plot[i,:] = (U_k + u_s).T

        ## Simulation of system
        if   i > int(np.round(5*N/7)) and i <= int(np.round(6*N/7)):
            x_k = test_model.simulation_step( dt, U_plot[i,:] + w_k.T + disturbance.T )
        else:
            x_k = test_model.simulation_step( dt, U_plot[i,:] + w_k.T )

        y_k = test_model.measurement( x_k ) + v_k # Actual read value of real system
        z_k = test_model.output( x_k )

        # MPC TOOLBOX - Compute optimal input
        Y_k   = matrix(y_k) - matrix(y_s)
        U_k, U_pred, z, Z_pred, X_k_k, W_k_k = mpc_compute( Y_k, X_k_k, W_k_k, U_k, Phi_x, Phi_w, Gamma, \
            U_min, U_max, du_max, soft_bounds, r_, n, Q_z, Q_du, Q_sl, Q_su, Q_tl, Q_tu, \
            P, A, B, G, C, Q, R, S, constraints=constraints )

        ## PLOT - Store results
        T_plot[i+1]  = time.time() - T_0
        Y_plot[i,:]  = np.array(Y_k + matrix(y_s)).reshape((1,ny))
        R_plot[i,:]  = np.array(matrix(r)).reshape((1,nZ))

    ## Finalising outputs
    T_ = matrix( range(0,(N+1)*T_s, T_s) )
    T_plot += T_
    
    ## Return statement
    return T_plot, Y_plot, R_plot, U_plot, plot_info

def closed_loop_plot( T_plot, Y_plot, R_plot, U_plot, plot_info ):
    ## General plot information
    T_s    = plot_info['T_s']
    u_min  = plot_info['u_min']
    u_max  = plot_info['u_max']
    N      = T_plot.size[0]
    
    _, ny = Y_plot.size
    _, nu = U_plot.size
    _, nr = R_plot.size

    ## Plot outputs: Initialisation
    fig1, ax1 = plt.subplots( ncols=nr, nrows=nr )  # Initialise subplots
    if ny == 1:
        ax1 = [ax1]
    fig1.suptitle( 'Output timeline', fontsize=15 )

    ## Plot outputs: Z_1, ..., Z_nz
    pl = [None] * 2                     # Initialise axes
    n1 = len(ax1); n2 = len(ax1[0])
    for i in range(n1):
        for j in range(n2):
            label = 'Measurement: ($Y_%d$)' % (i+j+1)
            name    = '$Y_%d$' % (i+1)
            tmp     = matrix( [np.nan, Y_plot[:,i]] )
            pl[0],  = ax1[j,i].plot( T_plot, tmp, 'c-', label=label )

        
        tmp = matrix( [np.nan, R_plot[:,i]] )
        pl[1], = ax1[j,i].step( T_plot, tmp, color='red', linestyle='--', label='Desired level'  )
        ax1[j,i].set_ylim(-1, 48.5)
        ax1[j,i].legend( handles=pl, fontsize=15 )
        ax1[j,i].legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15 ) # Place legend outside plot
        ax1[j,i].set_ylabel(name)
        ax1[j,i].set_xlabel('Time (minutes)')

    ## Plot inputs: Initialisation
    ax2 = [None] * nu                   # Initialise axes
    fig2, ax2 = plt.subplots(nrows=nu)  # Initialise subplots
    if nu == 1:
        ax2 = [ax2]
    fig2.suptitle( 'Input timeline', fontsize=15 )

    ## Plot input: U_1
    p2 = [None] * 2                     # Initialise axes
    for i in range(nu):
        name  = '$U_%d$' % (i+1)
        label = 'Input: %s' % name
        bound_interval = u_max[i]-u_min[i]     # Define bound interval
        
        tmp = matrix( [U_plot[:,i], np.nan] )
        p2[0], = ax2[i].step( T_plot, tmp , 'b', label=label  )
        p2[1], = ax2[i].plot( T_plot, u_min[i]+matrix(0.0,(N,1)), 'r--', label='Bounds' )
        _,     = ax2[i].plot( T_plot, u_max[i]+matrix(0.0,(N,1)), 'r--' )
        ax2[i].set_ylim( u_min[i]-0.1*bound_interval, u_max[i]+0.1*bound_interval )
        ax2[i].legend( handles=[ p2[0], p2[1] ], fontsize=15 )
        ax2[i].legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15 ) # Place legend outside plot
        ax2[i].set_ylabel(name)
        ax2[i].set_xlabel('Time (minutes)')

    ## Show plots
    #plt.tight_layout()  # Ensure x-ticks aren't cut off
    plt.show()
    
    ## Return statement
    return fig1, ax1, fig2, ax2