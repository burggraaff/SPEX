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


    def refractive_index(self, wavelengths, temperature):
        """
        Calculate the refractive index at given wavelengths and temperature using
        the Sellmeier equation.

        Sellmeier equation: N[lambda] = SQRT(P1 + P2*lambda^2 / (lambda^2 - P3) $
                                         + P4*lambda^2 / (lambda^2 - P5) $
                                         + ...
                                         + P(N-1)*lambda^2 / (lambda^2 - PN) )

        The resulting array will have a shape of [nr_wavelengths, 2] where the last
        axis represents the ordinary (0) and extraordinary (1) refractive indices.
        The array will consist of complex numbers.
        """
        # Convert from nm to microns
        wavelengths_um = wavelengths / 1000.

        # Extract the Sellmeier coefficients from the data - odd and even elements, after the first
        B = self.data_n[:,1::2]
        C = self.data_n[:,2::2]

        # Calculate the individual terms first
        wavelengths_squared = wavelengths_um[:,np.newaxis,np.newaxis]**2
        BC_terms = B*wavelengths_squared / (wavelengths_squared - C)

        # Calculate the refractive index
        n = np.sqrt(self.data_n[:,0] + np.nansum(BC_terms, axis=2))

        # Temperature correction respective to reference temperature
        dT = temperature - self.data_dndT[:,0]

        if self.data_dndT[0,7] == 0:
            # Schott's Sellmeier-type equation
            # https://www.schott.com/d/advanced_optics/02ffdb0d-00a6-408f-84a5-19de56652849/1.2/tie_29_refractive_index_and_dispersion_eng.pdf
            dn_abs_dT = (n**2 - 1) / (2*n) * (self.data_dndT[:,1] * dT + self.data_dndT[:,2] * dT**2 + self.data_dndT[:,3] * dT**3 + (self.data_dndT[:,4] * dT + self.data_dndT[:,5] * dT**2) / (wavelengths_um[:,np.newaxis]**2 - self.data_dndT[:,6]**2))
        else:
            # Ghosh dndT equation
            # https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-7-1540 ?
            Econv = 1e9 * 6.626e-34 * 2.9979e8 / 1.6e-19
            wavelengths_ig = Econv / self.data_dndT[:,4]
            RoRe = wavelengths_um**2 / (wavelengths_um**2 - wavelengths_ig[:,np.newaxis]**2)
            dn_abs_dT = (self.data_dndT[:,2,np.newaxis] * RoRe + self.data_dndT[:,3,np.newaxis] * RoRe**2).T / (2*n)

        n += dn_abs_dT * dT

        return n


    def retardance(self, wavelengths, temperature):
        """
        Calculate the retardance of this material at given wavelengths and temperature.
        """
        pass


Al2O3 = Material(name="Al2O3/Sapphire",
                data_n = np.array([[1., 1.43134936, 0.0726631**2, 0.65054713, 0.1193242**2, 5.3414021, 18.028251**2],
                                   [1., 1.5039759,  0.0740288**2, 0.55069141, 0.1216529**2,6.59273791, 20.072248**2]]),
                data_dndT = np.array([[293., 1.755, -45.2665E-06, 83.5457E-06, 8.27, 5.85,  7.21, -2.4],
                                      [293., 1.748, -39.8961E-06, 81.9579E-06, 8.00, 5.42,  6.47, -2.2]]),
                data_transmission = None,
                data_TEC = np.array([7.21e-6, 6.47e-6]),
                source = "Malitson & Dodge 1972, J. Opt. Soc. Am. 62, 1405")


MgF2 = Material(name="MgF2",
                data_n = np.array([[1., 0.48755108, 0.04338408**2, 0.39875031, 0.09461442**2, 2.3120353, 23.793604**2],
                                   [1., 0.41344023, 0.03684262**2, 0.50497499, 0.09076162**2, 2.4904862, 23.771995**2]]),
                data_dndT = np.array([[293., 1.290, -37.2043e-06, 39.3186e-06, 13.10, 8.00,  9.3, -4.70],
                                      [293., 1.290, -56.7859E-06, 57.3986E-06, 15.50, 8.00, 14.2, -6.90]]),
                data_transmission = None,
                data_TEC = np.array([9.3e-6, 14.2e-6]),
                source = "Dodge 1984, Applt. Opt. 23")

SiO2 = Material(name="SiO2/Quartz",
                data_n = np.array([[1.28604141, 1.07044083, 1.00585997*0.01, 1.10202242, 100.],
                                   [1.28851804, 1.09509924, 1.02101864*0.01, 1.15662475, 100.]]),
                data_dndT = np.array([[293., 1.515, -61.184E-06,  43.9990E-06, 10.30, 8.90,  6.88, -3.02],
                                      [293., 1.520, -70.1182E-06, 49.2875E-06, 10.30, 8.90, 12.38, -3.32]]),
                data_transmission = None,
                data_TEC = np.array([6.88e-6, 12.38e-6]),
                source = "Ghosh 1999 [https://doi.org/10.1016/S0030-4018(99)00091-7]")

# ; beta-BaB2O4, BBO, 0.22 - 1.06 micron - dn/dT values of beta-BBO
# ; Eimerl, D., Davis, L., and Velsko, S., Optical, mechanical, and thermal properties of barium borate,
# ; J. Appl. Phys. 62, 1968 (1987).
# ; Ghosh, G, Temperature dispersion of refractive indices in βBaB2O4 and LiB3O5 crystals for nonlinear optical devices
# ; J. Appl. Phys. 78, 6752 (1995)
# optics[1].N[0:4,0] = [1., 1.73651, 0.0120649, 0.0758505, 264.897]
# optics[1].N[0:4,1] = [1., 1.36847, 0.0100179,   1.29495, 187.560]
# optics[1].dNdT[*,0] = [293., 1.610, -19.3007E-6, -34.9683E-6, 19.00, 6.43,  4.00,  1.40]
# optics[1].dNdT[*,1] = [293., 1.520, -141.421E-6, 110.8630E-6, 17.00, 6.43, 36.00, -5.40]
# optics[1].tec = 1E-6*[4.00, 36.00]

# ; CaCO3, Calcite, 0.2 - 2.2 micron
# ; Gray, D. E., (Ed.), American Institute of Physics Handbook, 3rd ed. (McGraw-Hill, New York, 1972).
# optics[2].N[0:8,0] = [1., 0.8559, 0.0588^2, 0.83913, 0.141^2, 0.0009, 0.197^2, 0.6845, 7.005^2]
# optics[2].N[0:6,1] = [1., 1.0856, 0.07897^2, 0.0988, 0.142^2, 0.317, 1.468^2]
# optics[2].dNdT[*,0] = [293., 1.613, -121.689E-06, 122.494E-06, 10.80, 10.00, 25.00, -7.60]
# optics[2].dNdT[*,1] = [293., 1.472,  12.7011E-06, 20.4803E-06,  9.05,  6.83, -3.70, -1.20]
# optics[2].tec = 1E-6*[25.00, -3.70]

# ; TiO2, Rutile, 0.43 - 1.5 micron
# ; DeVore,J. R., Refractive index of rutile and sphalerite,
# ; J. Opt. Soc. Am. 41, 416 (1951).
# optics[5].N[0:4,0] = [1., 4.913, 0.0, 0.2441, 0.0803]
# optics[5].N[0:4,1] = [1., 6.097, 0.0, 0.3322, 0.0843]
# optics[5].dNdT[*,0] = [293., 2.432, -132.253E-06, 64.5269E-06, 4.10, 3.50,  8.98, -0.46]
# optics[5].dNdT[*,1] = [293., 2.683, -127.565E-06, 45.2141E-06, 4.10, 3.50,  6.87, -0.26]
# optics[5].tec = 1E-6*[8.98, 6.87]

# ; YVO4, 0.5 - 1.06 micron, HoOM Ref 112/113 - dn/dT values of TiO2!
# ; Maunder, E. A. and DeShazer, L. G., Use of yttrium orthovanadate for ; polarizers,
# ; J. Opt. Soc. Am. 61, 684A (1971).
# ; Lomheim, T. S. and DeShazer, L. G., Optical absorption intensities
# ; of trivalent neodymium in the uniaxial crystal yttrium orthovanadate,
# ; J. Appl. Phys. 49, 5517 (1978).
# optics[6].N[0:2,0] = [1., 2.7665, 0.026884]
# optics[6].N[0:2,1] = [1., 3.5930, 0.032103]
# optics[6].dNdT[*,0] = [293., 2.432, -132.253E-06, 64.5269E-06, 4.10, 3.50,  8.98, -0.46]
# optics[6].dNdT[*,0] = [293., 2.683, -127.565E-06, 45.2141E-06, 4.10, 3.50,  6.87, -0.26]
# optics[6].tec = 1E-6*[8.98, 6.87]

# ; Artificial material with spectrally constant refractive index of 1.5 and birefringence of 0.01
# optics[9].N[0,0] = [2.25]
# optics[9].N[0,1] = [2.2801]
# optics[9].dNdT[*,0] = [293., 1.5, 0.0E-06, 0.0E-06, 4.10, 3.50,  8.98, -0.46]
# optics[9].dNdT[*,0] = [293., 1.51, 0.0E-06, 0.0E-06, 4.10, 3.50,  6.87, -0.26]
# optics[9].tec = 1E-6*[0.0, 0.0]
