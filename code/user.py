import numpy as np
from simulation import AFMSimulation
from cantilever import Cantilever, Tip


###### Define the used tip ######
my_tip = Tip(diameter= 0)

###### Define the used cantilever ######
my_cantilever = Cantilever( volume_density    = 3170,     #[kg/m^3]
                            natural_frequency = 350e3,    #[Hz]
                            stiffness         = 42,       #[N/m]
                            beam_depth        = 30e-6,    #[m]
                            beam_thickness    = 7.5e-6,   #[m]
                            beam_length       = 1e-6,     #[m]
                            tip               = my_tip,   
                            tip_distance      = 0)


###### Set up the simulation ######
my_simulation = AFMSimulation( cantilever                = my_cantilever,   
                               surface_Hamaker_constant  = 16.70e-20,        #[J]
                               initial_height            = 2.5e-5,           #[m]
                               approach_velocity         = 1e-2,             #[m/s]
                               inclination_angle         = np.pi*(11/180),   #[rad]
                               delta_chi                 = 0.01,             #Spatial grid size
                               delta_tau                 = 1e-3,             #Temporal grid size
                               n                         =  1e6)             #Maximum number of nodes in time


my_simulation.plot_initial_deflection()
my_simulation.run()
my_simulation.plot_tip_height()
my_simulation.plot_beam_deflection()
print(my_simulation.get_snap_in_time())