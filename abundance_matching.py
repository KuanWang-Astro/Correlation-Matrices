import numpy as np
from halotools.mock_observables import return_xyz_formatted_array

class AM(object):
    def __init__(self,haloprop,halotable,Lbox,lum_threshold):
        self.haloprop = haloprop
        self.propcol = halotable[haloprop]
        self.Lbox = Lbox
        self.lum_threshold = lum_threshold
        self.halopos = return_xyz_formatted_array(halotable['halo_x'], halotable['halo_y'], halotable['halo_z'],\
            velocity=halotable['halo_vz'], velocity_distortion_dimension='z',period=Lbox)  ##distorted
        
    def calc_Ngal(self):
        self.Ngal = int((self.Lbox**3)*self.ngal())
        
    def ngal(self):
        if self.lum_threshold == -22:
            return 0.00005
        elif self.lum_threshold == -21.5:
            return 0.00028
        elif self.lum_threshold == -21.0:
            return 0.00116
        elif self.lum_threshold == -20.5:
            return 0.00318
        elif self.lum_threshold == -20.0:
            return 0.00656
    
    def match_gal(self):
        self.calc_Ngal()
        self.occupied_halo_idx = np.argsort(-self.propcol)[:self.Ngal]
        self.galpos = self.halopos[self.occupied_halo_idx]
        return self.galpos
    