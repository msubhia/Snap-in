"""
MIT Multifunctional Metamaterials (META) Lab
UROP Spring 2024, Subhi Abordan (msubhi_a@mit.edu)
"""

import numpy  as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cantilever import Cantilever

from fdm import *


class AFMSimulation(object):
    """
    Simulation of the mirco-cantilevers dynamics in Atomic Force Microscopes (AFMs) experiments.
    The simulation models the cantilever using the Euler's beam equation and solve for the dynamics
    numerically using the finite difference method (FDM).
    """

    def __init__(self, cantilever, surface_Hamaker_constant, initial_height, approach_velocity, inclination_angle, delta_chi = 0.01, delta_tau= 1e-3, n =  1e6):
        """
        Constructor for AFMSimulation class instances, inputs required:

                cantilever:                     the cantilever object used in the experiment, of type Cantilever.
                surface_Hamaker_constant: [J]   Hamaker constant that models the force between the sample surface and the cantilever object.
                initial_height: [m]             the initial height of the cantilever from the sample at the beginning of the experiment.
                approach_velocity: [m/s]        the approach velocity of the cantilever while it moves closer to the sample, assumer constant.
                inclination_angle: [rad]        the inclination angle of the cantilever.
                delta_chi:                      Spatial grid size (along the beam), so non-dimentionlized chi will range from 0 -> 1.
                delta_tau:                      Temporal grid size, so non-dimentionlized tau will range from 0 -> 1.
                n:                              Maximum number of nodes in time (maximum time) 
        """

        self.cantilever = cantilever
        self.surface_Hamaker_constant = surface_Hamaker_constant
        self.initial_height = initial_height
        self.approach_velocity = approach_velocity
        self.inclination_angle = inclination_angle
        self.delta_chi = delta_chi
        self.delta_tau = delta_tau
        self.n = n

        self.results = None
        self.initial_deflection = None
        self.stress = None
        self.snap_in_time = None







    def get_theotical_snap_distance(self):
        """
        Returns the theotical snap-in distance in [m] using the expression delta_0 = (A*b*cos(theta)/(8*pi*k))^(1/4) 
        """
        return np.power((self.surface_Hamaker_constant * self.cantilever.beam_depth * np.cos(self.inclination_angle)) / (8 + np.pi * self.cantilever.stiffness), 1/4)






    def get_alpha(self):
        """
        Returns the dimensionless parameter alpha in the beam equation using α=(L^4 ω_0^2 ρ)/EI
        """
        cantilever = self.cantilever
        return ((cantilever.beam_length**4) * cantilever.linear_density * (cantilever.natural_frequency**2)) / (cantilever.young_modulus * cantilever.second_moment_area)






    def get_beta(self):
        """
        Returns the ratio of bulk-surface force to restoring force of the canteliver, beta defined in the model as
            β=(L^4 Ab cos⁡θ)/(24πl^4 EI)
        """ 
    
        ## BETA NOT GENERAL, DEPENDS OF THE SHAPE AND FORCE LATER
        cantilever = self.cantilever
    
        num   = (cantilever.beam_length**4) * self.surface_Hamaker_constant * (cantilever.beam_depth) * (np.cos(self.inclination_angle))
        denum = 24 * np.pi * (self.get_theotical_snap_distance()**4) * cantilever.young_modulus * cantilever.second_moment_area

        return num/denum






    def generate_initial_deflection(self):
        """
        Generates the initial deflection of the beam given the initial conditions of the simulation.
        TO BE USED as an initial deflection in the simulation.

        Return:
        Array of the values of υ(χ), at χ between 0->1 with self.delta_chi interval, representing the initial deflection of the beam
        """

        delta_0 = self.get_theotical_snap_distance()                                                            # The theortical approximation to the snap-in distance (see paper)

        zeta_0 = self.initial_height/delta_0                                                                    # Dimensionless initial height
        fract = (self.cantilever.beam_length/ delta_0) * np.sin(self.inclination_angle) * self.delta_chi

        self.initial_deflection = static_beam_bvp(beta = self.get_beta(),
                                             zeta_0= zeta_0,
                                             kappa = fract,
                                             delta_chi= self.delta_chi)

        return self.initial_deflection







    def run(self):
        """
        Runs the simulation given the experiment enviroment specified by: cantilever, surface_Hamaker_constant, 
        initial_height, approach_velocity, inclination_angle. For more information about the non-dimenalization,
        please read own provided paper.

        Args:
            delta_chi:              Spatial grid size (along the beam), so non-dimentionlized chi will range from 0 -> 1.
            delta_tau:              Temporal grid size, so non-dimentionlized tau will range from 0 -> 1.
            n:                      Maximum number of nodes in time (maximum time)            
        """

        delta_0 = self.get_theotical_snap_distance()                                                            # The theortical approximation to the snap-in distance (see paper)
        zeta_0 = self.initial_height/delta_0                                                                    # Dimensionless initial height
        nu = self.approach_velocity/(delta_0*self.cantilever.natural_frequency)                                 # Dimensionless approach velocity
        fract = (self.cantilever.beam_length/ delta_0) * np.sin(self.inclination_angle) * self.delta_chi

        self.results = dynamic_beam_fdm( alpha = self.get_alpha(),
                            beta = self.get_beta(),
                            zeta_0 = zeta_0,
                            nu = nu,
                            kappa = fract,
                            initial_deflection= None,
                            delta_chi = self.delta_chi,
                            delta_tau = self.delta_tau,
                            n = self.n,
                            lambda_ = 0.50)

        return self.results
    




    def compute_stress(self, plot:bool = False):
        """
        Gives estimate for the stress across the cantilever body at the end of the simulation. Calculated using FDM, as the second derivative of the deflection. 
        """

        if self.results is None:
            self.run()

        delta_0 = self.get_theotical_snap_distance()  
        print(delta_0)

        upsilon_trunc = self.results["upsilon_trunc"]
        stress = - (self.cantilever.beam_thickness/2) * self.cantilever.young_modulus * get_second_derevative(upsilon_trunc[-1, :], delta_unit = self.delta_chi)

        #T DELTA0 /l^2
        stress_new =  - (self.cantilever.beam_thickness/2) * self.cantilever.young_modulus * get_first_derevative(get_first_derevative(upsilon_trunc[-1, :] , delta_unit = self.delta_chi), delta_unit = self.delta_chi)


        if plot:
            m = int(1/self.delta_chi)
            chi = np.linspace(0, 1, m+1)
            plt.plot(chi, stress, label = "old")
            # plt.plot(chi, stress_new, label = "new")
            plt.xlabel("x/L")
            plt.ylabel("stress")
            plt.title("The stress across the cantilever body at the end of the simulation")
            plt.grid(True)
            plt.legend()
            plt.show()

        return stress






    def get_snap_in_time(self):
        """
        Returns the snap in time
        """

        if self.results is None:
            self.run()

        upsilon_trunc = self.results["zeta_trunc"] - self.results["upsilon_trunc"]
        zeta_trunc = self.results["zeta_trunc"] 

        tb = 0
        for i in range(len(upsilon_trunc)):
            if upsilon_trunc[i]<zeta_trunc[-1]:
                tb = i
                break

        self.snap_in_time = (len(zeta_trunc) - tb) * self.delta_tau
        return self.snap_in_time








    def plot_tip_height(self, offset = 100):
        """
            Plots the cantilever tip height form the sample surface as a function of time after running the simulation.
        """

        if self.results is None:
            self.run()

        delta_0 = self.get_theotical_snap_distance()                                                            # The theortical approximation to the snap-in distance (see paper)
        fract = (self.cantilever.beam_length/ delta_0) * np.sin(self.inclination_angle) * self.delta_chi

        upsilon_trunc = self.results["upsilon_trunc"]
        tau_trunc = self.results["tau_trunc"]
        zeta_trunc = self.results["zeta_trunc"]
        H = zeta_trunc - upsilon_trunc[:, -1] #- fract  * int(1/self.delta_chi)

        plt.plot(tau_trunc[-offset:], H[-offset:], label = "H")
        plt.plot(tau_trunc[-offset:], zeta_trunc[-offset:], label = "zeta")
        plt.xlabel("\tau")
        plt.ylabel("Dimensionless Height")
        plt.legend()
        plt.title("Tip Height form the surface")
        plt.grid(True)
        plt.show()







    def plot_beam_deflection(self):
        """
        Plots the beam deflection and shape at different times in during the simulation.
        """

        if self.results is None:
            self.run()

 
        upsilon_trunc = self.results["upsilon_trunc"]
        trunc = self.results["trunc"]
        zeta_trunc = self.results["zeta_trunc"]
        chi = self.results["chi"]
        tau_trunc = self.results["tau_trunc"]

        # Compute y values for plotting
        indices = range(trunc, trunc - 1500, -100)
        y_s = [zeta_trunc[i] - upsilon_trunc[i, :] for i in indices]

        # Define colors and colormap for better plotting 
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        color_steps = len(y_s)
        cmap = LinearSegmentedColormap.from_list("custom_spectrum", colors, N=color_steps)

        plt.figure(figsize=(12, 8))

        for y, color in zip(y_s, [cmap(i) for i in np.linspace(0, 1, color_steps)]):
            plt.plot(chi, y, color=color, linewidth=2)

        plt.xlabel(r'$x/L$', fontsize=14)
        plt.ylabel(r'$(z-u)/\delta_0$', fontsize=14)
        plt.title('Deflections of the Beam at Different Times', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Enhance the legend with formatted time labels
        legend_labels = [f'$t / \omega_0^{{-1}} = {tau_trunc[i]:.2f}$' for i in indices]
        plt.legend(legend_labels, loc='upper left', fontsize=12)

        plt.tight_layout()
        plt.show()








    def plot_initial_deflection(self):
        """
        Plots the beam deflection and shape at t=0.
        """

        if self.initial_deflection is None:
            self.generate_initial_deflection()

        m = int(1/self.delta_chi)  # Number of nodes in space
        chi = np.linspace(0, 1, m+1)  # Spatial grid

        plt.figure(figsize=(10, 6))
        plt.plot(chi, -self.initial_deflection, color='red', linewidth=3, label='Beam Deflection')
        plt.xlabel(r'$x/L$', fontsize=14)
        plt.ylabel(r'$(z-u)/\delta_0$', fontsize=14)
        plt.title('Initial Deflection of the Beam', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=1)  # Add a horizontal line at y=0
        plt.axvline(0, color='black', linewidth=1)  # Add a vertical line at x=0
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.show()
