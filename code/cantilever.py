"""
MIT Multifunctional Metamaterials (META) Lab
UROP Spring 2024, Subhi Abordan (msubhi_a@mit.edu)
"""

class Cantilever(object):
    """
    Rectangualy-shaped Atomic Force Microscope (AFM) Cantilever.
    """

    def __init__(self, volume_density, natural_frequency, stiffness, beam_depth, beam_thickness, beam_length, tip = None, tip_distance = None):
        """
        Constructor for the Cantilever class, given fundamentals information of the Cantilever as inputs:
            - volume_density: [kg/m^3]
            - natural_frequency: [Hz]
            - stiffness: [N/m]
            - beam_depth: [m]
            - beam_thickness: [m]
            - beam_length: [m]
            - tip: the tip of the cantilever
            - tip_distance: [m] the distance from the end of the cantilever to the tip

        It create a Cantilever object with these properties in addition to: second_moment_area, linear_density, young_modulus
        all calculated from the given arguments using their analytical expressions.
        """

        ####  Given as Arguments ####
        self.volume_density = volume_density
        self.natural_frequency = natural_frequency
        self.stiffness = stiffness
        self.beam_depth = beam_depth
        self.beam_thickness = beam_thickness
        self.beam_length = beam_length
        self.tip = tip                              # could be None if no tip
        self.tip_distance = tip_distance            # could be None if no tip

        ####  Calculated From Arguments ####
        self.second_moment_area= (1/12) * beam_depth * (beam_thickness**3)                      # analtical form for rectangular shapes
        self.linear_density = volume_density * (beam_depth * beam_thickness)                    # mu = rho * cross section area
        self.young_modulus = stiffness * (beam_length**3) * (1/(3*self.second_moment_area))     # equation in Simo's code



class Tip(object):
    """
    Spherical tips for AFM Cantilevers
    """

    def __init__(self, diameter):
        """
        diameter in [m]
        """
        self.diameter = diameter

    
