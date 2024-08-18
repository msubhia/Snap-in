# Predicting the Dynamics of Microcantilevers for Radiative Heat and Momentum Transfer on the Nanoscale | Snap-in

## Overview
In the field of radiative heat transfer, the near-field regime occurs when the sizes of objects and/or the distances between them are smaller than the peak wavelength of thermal radiation, according to Wien’s law. The deviation from traditional laws of thermal radiation in the near-field is well-documented; however, the transition to the contact regime—from radiation to conduction—is poorly understood. This transition is often studied using scanning probe microscopy, which includes various techniques that utilize microcantilevers as probes, such as atomic force microscopy (AFM). A significant challenge in these studies is overcoming “snap-in” — the abrupt jump to contact between two closely spaced surfaces due to surface forces.

A cantilever with a finite stiffness \(k\) approaches a sample surface. When the surface force gradient \(\nabla P\) exceeds the cantilever's stiffness \(k\), the cantilever beam becomes mechanically unstable, causing it to rapidly deflect towards the surface until contact is made. As shown in the figure:

<img width="763" alt="Screenshot 2024-08-17 at 9 12 00 PM" src="https://github.mit.edu/storage/user/26737/files/81ce1f32-7bf0-458a-af93-07e1d8904d68">

In our work, we model the snap-in phenomenon using the Euler-Bernoulli beam equation and build a simulation based on this model by solving the equation numerically using the Crank–Nicolson Finite Difference Method (FDM). We provide estimates for the snap-in dynamics in different AFM experiment setups, specifically the snap-in time and the snap-in distance, defined as the time and distance from when the beam loses stability until it makes contact with the sample.

<br><br>

## Code

- **[cantilever.py](https://github.mit.edu/msubhi-a/snap-in/blob/master/code/cantilever.py)** contains the class for the cantilever object, which represents an Atomic Force Microscope (AFM) microcantilever with its physical properties.

- **[fdm.py](https://github.mit.edu/msubhi-a/snap-in/blob/master/code/fdm.py)** contains the implementation of the mathematical numerical methods used in the simulation, highlighted by the FDM functions that solve the Euler-Bernoulli beam equation in both dynamic and static cases.

- **[simulation.py](https://github.mit.edu/msubhi-a/snap-in/blob/master/code/simulation.py)** contains the simulation code that builds on the mathematical utility functions implemented in `fdm.py`. It is instantiated with initial conditions from AFM experiments and provides attributes to generate initial beam deflection, run the simulation, compute the stress along the beam, and determine the snap-in time. Additionally, it offers functions to plot the cantilever tip height as a function of time during the snap-in, plot beam deflection, and plot initial deflection.

For more information about the simulation, please refer to the function docs in the code or the project report/paper.

<br><br>

## Getting Started
The following script **[user.py](https://github.mit.edu/msubhi-a/snap-in/blob/master/code/user.py)** demonstrates how to use the simulation by initializing the cantilever object and setting up the simulation environment.

```python
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
                               n                         = 1e6)              #Maximum number of nodes in time

my_simulation.plot_initial_deflection()
my_simulation.run()
my_simulation.plot_tip_height()
my_simulation.plot_beam_deflection()
print(my_simulation.get_snap_in_time())
```
<br><br>

The following is the results of running the above script

![initial_deflection](https://github.mit.edu/storage/user/26737/files/8567e4f0-0859-4c4f-a477-ed7ad72cd82e)
![tip_height](https://github.mit.edu/storage/user/26737/files/261b86dc-d91b-4f4b-a334-764847461c8b)
![defs](https://github.mit.edu/storage/user/26737/files/79a10c3d-4328-4930-a38b-8b1f96207fd4)



