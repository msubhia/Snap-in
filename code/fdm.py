"""
MIT Multifunctional Metamaterials (META) Lab
UROP Spring 2024, Subhi Abordan (msubhi_a@mit.edu)
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_bvp



def dynamic_beam_fdm(alpha, beta, zeta_0, nu, kappa, initial_deflection = None, delta_chi = 0.01, delta_tau = 1e-3, n = 1e6, lambda_ = 0.50):
    """
    Solves the PDE beam equation using Finite Difference Method (FDM).
    Equation of the form:

                (∂^4 υ)/(∂χ^4 )= -α (∂^2 υ)/(∂τ^2 )+β 1/(ζ-υ- kappa * χ⁡)^3 

    For boundry conditions:
        υ(0) = 0
        υ'(0) = 0
        υ''(1) = 0
        υ'''(1) = 0

    Args:
        delta_chi:              Spatial grid size (along the beam), so non-dimentionlized chi will range from 0 -> 1.
        delta_tau:              Temporal grid size, so non-dimentionlized tau will range from 0 -> 1.
        n:                      Maximum number of nodes in time (maximum time)  
        lambda_:                Weight parameter in FDM
        alpha:                  Parameter alpha in dimensionless equ
        beta:                   Parameter Beta in dimensionless equ
        zeta_0:                 Dimensionless initial height
        nu:                     Dimensionless approach velocity
    """

    n = int(n)
    r = delta_tau/(delta_chi**2)                    # Grid size ratio
    m = int(1/delta_chi)                            # Number of nodes in space

    chi = np.linspace(0, 1, m+1)                    # Spatial grid
    tau = np.linspace(0, n*delta_tau, n+1)          # Temporal grid

    upsilon = np.zeros((n+1,m+1))                   # The dimensionless deflection array
                                                    # upsilon as a function of chi goes across rows
                                                    # upsilon as a function of tau goes down columns

    zeta = zeta_0 - nu * tau                                                          # Generate driven height

    ## TO COMMENT AND DEBUG
    fract = kappa

    ################## Generate and solve the system of equations at each time step ##################

    ########### initial conditions: time_step = 0, 1
    for i in range(len(chi)):
        if initial_deflection is None:
            upsilon[0][i] = 0                               # Initial displacement: upsilon(x, 0) = f(x)
        else:
            upsilon[0][i] = initial_deflection[i]

        upsilon[1][i] = upsilon[0][i]                   # Initial velocity: d(upsilon)/dt|t=0 = 0

    print("Starting Simultion... ")
    ########### At later time steps: time_step > 1
    for time_step in range(2, n+2):

        LHS = np.zeros((m+1, m+1))
        RHS = np.zeros(m+1)

        ## Special rows (boundary conditions)
        # i = 2
        LHS[2][2] =  6*lambda_ + alpha/(r**2)
        LHS[2][3] = -4*lambda_
        LHS[2][4] =  lambda_

        RHS[2]    = (2 * (alpha/(r**2))) * upsilon[time_step-1][2]                                                               \
                    + beta * (delta_chi**4)/ np.power((zeta[time_step-1] - upsilon[time_step-1][2] - 2 * fract),3)              \
                    - (1-lambda_) *  ( (6 + alpha/((r**2) * (1 - lambda_))) * upsilon[time_step-2][2] - 4 *  upsilon[time_step-2][3] + upsilon[time_step-2][4])
        
        # i = 3
        LHS[3][2] = -4*lambda_
        LHS[3][3] =  6*lambda_ + alpha/(r**2)
        LHS[3][4] = -4*lambda_
        LHS[3][5] =  lambda_

        RHS[3]    = (2 * alpha/(r**2)) * upsilon[time_step-1][3]                                                                \
                    + beta * (delta_chi**4)/ np.power((zeta[time_step-1] - upsilon[time_step-1][3] - 3 * fract),3)              \
                    - (1-lambda_) *  ( - 4 *  upsilon[time_step-2][2] + (6 + alpha/((r**2) * (1 - lambda_))) * upsilon[time_step-2][3] - 4 *  upsilon[time_step-2][4] + upsilon[time_step-2][5])
        
        # i = m-1
        LHS[m-1][m-3] = lambda_
        LHS[m-1][m-2] = -4 * lambda_
        LHS[m-1][m-1] = 5 * lambda_ + alpha/(r**2)
        LHS[m-1][m] = -2 * lambda_

        RHS[m-1] = (2 * alpha/(r**2)) * upsilon[time_step-1][m-1] \
                + beta * (delta_chi**4)/ np.power((zeta[time_step-1] - upsilon[time_step-1][m-1] - (m-1) * fract),3) \
                - (1-lambda_) * (upsilon[time_step-2][m-3] - 4*upsilon[time_step-2][m-2] + (5 + alpha/((r**2) * (1 - lambda_))) * upsilon[time_step-2][m-1] -2*upsilon[time_step-2][m])

        # i = m
        LHS[m][m-2] = 2* lambda_
        LHS[m][m-1] = -4 * lambda_
        LHS[m][m] = 2 * lambda_ + alpha/(r**2)

        RHS[m] = (2 * alpha/(r**2)) * upsilon[time_step-1][m] \
                + beta * (delta_chi**4)/ np.power((zeta[time_step-1] - upsilon[time_step-1][m] - m * fract),3) \
                - (1-lambda_) * (2*upsilon[time_step-2][m-2] - 4*upsilon[time_step-2][m-1] + (2 + alpha/((r**2) * (1 - lambda_))) * upsilon[time_step-2][m])

        #remaining rows
        for j in range(4, m-1):
            LHS[j][j-2] = lambda_
            LHS[j][j-1] = -4 * lambda_
            LHS[j][j] = 6 * lambda_ + alpha/(r**2)
            LHS[j][j+1] = -4 * lambda_
            LHS[j][j+2] = lambda_

            RHS[j] = (2 * alpha/(r**2)) * upsilon[time_step-1][j] \
                    + beta * (delta_chi**4)/ np.power((zeta[time_step-1] - upsilon[time_step-1][j] - j * fract),3) \
                    - (1-lambda_) * (upsilon[time_step-2][j-2] - 4* upsilon[time_step-2][j-1] + (6 + alpha/((r**2) * (1 - lambda_))) * upsilon[time_step-2][j] - 4*upsilon[time_step-2][j+1] + upsilon[time_step-2][j+2])


        LHS = LHS[2:, 2:]                                                   # Remove the first two rows and columns of zeros
        RHS = RHS[2:]

        LHS_sparse = csc_matrix(LHS)                                        # Convert LHS into a sparse matrix
        
        x = spsolve(LHS_sparse, RHS)                                        # Now solve LHS x = RHS for x
        upsilon[time_step][2:] = x                                          # This x is upsilon[time_step]

        break_cond = zeta[time_step]-upsilon[time_step][-2] - fract  *m     # when the end of the cantilevel touched the sample
        print("tip height = ", break_cond)
        results = None

        if break_cond <= 0:

            upsilon_trunc = upsilon[:time_step,:]
            tau_trunc = tau[:time_step]
            zeta_trunc = zeta[:time_step]
            trunc = time_step - 1

            results = {"upsilon_trunc": upsilon_trunc,
                            "tau_trunc": tau_trunc,
                            "zeta_trunc": zeta_trunc,
                            "trunc": trunc,
                            "chi": chi}

            if True:
                print("Finished Running the Simulation")
                print("break_cond = ", break_cond)
                print("time step = ", trunc)
                print("tau = ", tau_trunc[trunc])
                print("zeta_trunc", zeta_trunc)

            break

    return results



def static_beam_bvp(beta, zeta_0, kappa, delta_chi = 0.01):
    """
    Equation of the form:

                (∂^4 υ)/(∂χ^4 )= f(υ, χ)
    
    where f(υ, χ) = β / (ζ-υ- kappa * χ⁡)^3 

    For boundry conditions:
    υ(0) = 0
    υ'(0) = 0
    υ''(1) = 0
    υ'''(1) = 0

    Args:
        delta_chi:              Spatial grid size (along the beam), so non-dimentionlized chi will range from 0 -> 1.
        beta:                   Parameter Beta in dimensionless equ
        zeta_0:                 Dimensionless initial height

    Return:
        Array of the values of υ(χ), at χ between 0->1 with delta_chi interval, representing the initial deflection of the beam

    """

    ### converting the fourth-order differential equation into a system of first-order differential equations
    # Let: y = [y1, y2, y3, y4]
    # y1 = υ
    # y2 = υ'
    # y3 = υ''
    # y4 = υ'''
    #
    # Hence:
    #  y1' = y2
    #  y2' = y3
    #  y3' = y4
    #  y4' = f(x,y1)

    def f(x, y1):
        """
        The inhomo term in ODE
        """
        return beta/np.power(zeta_0 - y1 - kappa * x, 3)
    
    def derivatives(x, y):
        dy_dx = np.zeros_like(y)
        dy_dx[0] = y[1]            # y1' = y2
        dy_dx[1] = y[2]            # y2' = y3
        dy_dx[2] = y[3]            # y3' = y4
        dy_dx[3] = f(x, y[0])      # y4' = f(x, y1)

        return dy_dx

    def boundary_conditions(ya, yb):
        return [ya[0] - 0,    # y1(0) = 0
                ya[1] - 0,    # y2(0) = 0
                yb[2] - 0,    # y3(x_end) = 0
                yb[3] - 0]    # y4(x_end) = 0

    m = int(1/delta_chi)
    x_mesh = np.linspace(0, 1, m+1)
    y_guess = np.zeros((4, x_mesh.size))

    solution = solve_bvp(derivatives, boundary_conditions, x_mesh, y_guess)

    if solution.success:
        return solution.y[0]
    else:
        raise Exception('Unable to find the initial deflection of the beam')



def get_second_derevative(f, delta_unit):
    """
    Given an array of the values of a function f at points x_i for i = 1, 2, 3, 4 ...
    This function returns the second derivative estimates using FDM
    """

    f_double_prime = np.zeros(len(f))

    f_double_prime[-1] = (f[-1] - 2 * f[-2] + f[-3]) / (delta_unit**2)
    f_double_prime[-2] = (f[-2] - 2 * f[-3] + f[-4]) / (delta_unit**2)

    f_double_prime[0] = (f[0] - 2 * f[1] + f[2]) / (delta_unit**2)
    f_double_prime[1] = (f[1] - 2 * f[3] + f[3]) / (delta_unit**2)

    for i in range(2, len(f) - 2):
        f_double_prime[i] = (f[i-1] - 2 * f[i] + f[i+1]) / (delta_unit**2)

    return f_double_prime



def get_first_derevative(f, delta_unit):
    """
    Given an array of the values of a function f at points x_i for i = 1, 2, 3, 4 ...
    This function returns the first derivative estimates using FDM
    """

    f_prime = np.zeros(len(f))

    f_prime[0] = 0
    f_prime[-1] = 0
    
    for i in range(1, len(f) - 1):
        f_prime[i] = (f[i+1] - f[i-1]) / delta_unit

    return f_prime









##### DON'T USE #######
#######################

## This function is working on an approximation with kappa = 0  
## Otherwise, there is another implementation above
def static_beam_fdm(beta, zeta_0, kappa, delta_chi = 0.01):
    """
    Solves the static PDE beam equation using Finite Difference Method (FDM).
    Equation of the form:

                (∂^4 υ)/(∂χ^4 )= f(υ, χ)
    
    where f(υ, χ) = β / (ζ-υ- kappa * χ⁡)^3 

    For boundry conditions:
    υ(0) = 0
    υ'(0) = 0
    υ''(1) = 0
    υ'''(1) = 0

    Args:
        delta_chi:              Spatial grid size (along the beam), so non-dimentionlized chi will range from 0 -> 1.
        beta:                   Parameter Beta in dimensionless equ
        zeta_0:                 Dimensionless initial height

    Return:
        Array of the values of υ(χ), at χ between 0->1 with 0.01 interval, representing the initial deflection of the beam

    """

    m = int(1/delta_chi)                            # Number of nodes in space

    LHS = np.zeros((m+1, m+1))
    RHS = np.zeros(m+1)

    c1 = (delta_chi**4) * beta / (zeta_0**3)
    c2 = 3 * kappa * delta_chi

    ## Special rows (boundary conditions)
    # i = 2
    LHS[2][2] = (6 - 2 * c2)
    LHS[2][3] = -4
    LHS[2][4] = 1
    RHS[2]    = c1 * (1 + 2 * c2)

    # i = 3
    LHS[3][2] = -4
    LHS[3][3] = (6 - 3 * c2)
    LHS[3][4] = -4
    LHS[3][5] = 1
    RHS[3]    = c1 * (1 + 3 * c2)

    # i = m-1
    LHS[m-1][m-3] = 1
    LHS[m-1][m-2] = -4
    LHS[m-1][m-1] = (5 - (m-1) * c2)
    LHS[m-1][m] = -2
    RHS[m-1]    = c1 * (1 + (m-1) * c2)

    # i = m
    LHS[m][m-2] = 2
    LHS[m][m-1] = -4
    LHS[m][m] = (2 - m * c2)
    RHS[m]    = c1 * (1 + m * c2)
    
    #remaining rows
    for j in range(4, m-1):
        LHS[j][j-2] = 1
        LHS[j][j-1] = -4
        LHS[j][j] = 6 - j * c2
        LHS[j][j+1] = -4
        LHS[j][j+2] = 1
        RHS[j] = c1 * (1 + j * c2)

    LHS = LHS[2:, 2:]                                       # Remove the first two rows and columns of zeros
    RHS = RHS[2:]

    LHS_sparse = csc_matrix(LHS)                            # Convert LHS into a sparse matrix
    upsilon_0 = spsolve(LHS_sparse, RHS)                    # Now solve LHS x = RHS for x
    upsilon_0 = np.array([0,0,*upsilon_0])

    return upsilon_0
