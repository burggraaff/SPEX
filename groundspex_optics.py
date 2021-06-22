"""
Optics for groundSPEX
"""
import numpy as np

class Material(object):
    """
    Optical material.
    """
    def __init__(self, data_n, data_dndT, data_transmission, data_TEC=None, name="GenericMaterial", source="imagination"):
        """
        Create the object.
        data_n: Parameters describing complex refractive index: [2, any]
        dndT: Thermal optical coefficient: [2, 8]
        transmission: Transmission per wavelength: [nr_lambda]
        TEC: Thermal expansion coefficient: [2]

        name: Name.
        source: Source of data.
        """
        # Initialise properties
        self.data_n = data_n
        self.data_dndT = data_dndT
        self.data_transmission = data_transmission
        self.data_TEC = data_TEC
        self.name = name
        self.source = source


    def __repr__(self):
        """
        The string that gets printed to describe this object.
        """
        return f"Optical material {self.name}, based on data from {self.source}"


    def calculate_refractive_index(self, wavelengths, temperature):
        """
        Calculate the refractive index at given wavelengths and temperature using
        the Sellmeier equation.
        Sellmeier equation: N[lambda] = SQRT(P1 + P2*lambda^2 / (lambda^2 - P3) $
                                         + P4*lambda^2 / (lambda^2 - P5) $
                                         + ...
                                         + P(N-1)*lambda^2 / (lambda^2 - PN) )

        The resulting array will have a shape of [2, nr_wavelengths] where the first
        axis represents the ordinary (0) and extraordinary (1) refractive indices.
        The array will consist of complex numbers.
        """
        # Extract the Sellmeier coefficients from the data - odd and even elements, after the first
        B = self.data_n[:,1::2]
        C = self.data_n[:,2::2]

        # Calculate the individual terms first
        wavelengths_squared = wavelengths[:,np.newaxis,np.newaxis]**2
        BC_terms = B*wavelengths_squared / (wavelengths_squared - C)

        # Calculate the refractive index
        n = np.sqrt(self.data_n[:,0] + np.nansum(BC_terms, axis=2))

        # Flip the axes
        n = n.T

        return n


Al2O3 = Material(name="Al2O3/Sapphire",
                data_n = np.array([[1., 1.43134936, 0.0726631**2, 0.65054713, 0.1193242**2, 5.3414021, 18.028251**2], [1., 1.5039759, 0.0740288**2, 0.55069141, 0.1216529**2,6.59273791, 20.072248**2]]),
                data_dndT = np.array([[293., 1.755, -45.2665E-06, 83.5457E-06, 8.27, 5.85,  7.21, -2.4], [293., 1.748, -39.8961E-06, 81.9579E-06, 8.00, 5.42,  6.47, -2.2]]),
                data_transmission = None,
                data_TEC = np.array([7.21e-6, 6.47e-6]),
                source = "Malitson & Dodge 1972, J. Opt. Soc. Am. 62, 1405")


MgF2 = Material(name="MgF2",
                data_n = np.array([[1., 0.48755108, 0.04338408**2, 0.39875031, 0.09461442**2, 2.3120353, 23.793604**2], [1., 0.41344023, 0.03684262**2, 0.50497499, 0.09076162**2, 2.4904862, 23.771995**2]]),
                data_dndT = np.array([[293., 1.290, -37.2043e-06, 39.3186e-06, 13.10, 8.00,  9.3, -4.70], [293., 1.290, -56.7859E-06, 57.3986E-06, 15.50, 8.00, 14.2, -6.90]]),
                data_transmission = None,
                data_TEC = np.array([9.3e-6, 14.2e-6]),
                source = "Dodge 1984, Applt. Opt. 23")

SiO2 = Material(name="SiO2",
                data_n = np.array([[1.28604141, 1.07044083, 1.00585997*0.01, 1.10202242, 100.], [1.28851804, 1.09509924, 1.02101864*0.01, 1.15662475, 100.]]),
                data_dndT = np.array([[293., 1.515, -61.184E-06, 43.9990E-06, 10.30, 8.90,  6.88, -3.02], [293., 1.520, -70.1182E-06, 49.2875E-06, 10.30, 8.90, 12.38, -3.32]]),
                data_transmission = None,
                data_TEC = np.array([6.88e-6, 12.38e-6]),
                source = "Gosh 1999 (Halle)")
