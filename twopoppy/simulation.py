import dustpy as dp
import numpy as np
from simframe.frame import Field


class Simulation(dp.Simulation):

    __name__ = "TwoPopPy"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def initialize(self):
        '''Function initializes the simulation frame.

        Function sets all fields that are None with a standard value.
        If the grids are not set, it will call ``Simulation.makegrids()`` first.'''

        if not isinstance(self.grid.Nr, Field):
            self.makegrids()

        # Shapes needed for array initialization
        shape1 = (int(self.grid.Nr))
        shape1p1 = (int(self.grid.Nr)+1)
        shape2 = (int(self.grid.Nr), int(self.grid._Nm))
        shape2ravel = (int(self.grid.Nr*self.grid._Nm))
        shape2p1 = (int(self.grid.Nr)+1, int(self.grid._Nm))
        shape3 = (int(self.grid.Nr), int(
            self.grid._Nm), int(self.grid._Nm))

    def makegrids(self):
        '''Function creates radial grid.

        Notes
        -----
        The grids are set up with the parameters given in ``Simulation.ini``.
        If you want to have a custom radial grid you have to set the array of grid cell interfaces ``Simulation.grid.ri``,
        before calling ``Simulation.makegrids()``.'''

        # Number of mass species. Hard coded
        Nm = 2

        # The mass grid does not exist. But we store the size of the
        # particle dimension in a hidden variable.
        self.grid.addfield(
            "_Nm", Nm, description="# of particle species", constant=True)

        self._makeradialgrid()

    def _makeradialgrid(self):
        '''Function sets the mass grid using the parameters set in ``Simulation.ini``.'''
        if self.grid.ri is None:
            ri = np.logspace(np.log10(self.ini.grid.rmin), np.log10(
                self.ini.grid.rmax), num=self.ini.grid.Nr+1, base=10.)
            Nr = self.ini.grid.Nr
        else:
            ri = self.grid.ri
            Nr = ri.shape[0] - 1
        r = 0.5*(ri[:-1] + ri[1:])
        A = np.pi*(ri[1:]**2 - ri[:-1]**2)
        self.grid.addfield(
            "Nr", Nr, description="# of radial grid cells", constant=True)
        self.grid.addfield(
            "r", r, description="Radial grid cell centers [cm]", constant=True)
        self.grid.addfield(
            "ri", ri, description="Radial grid cell interfaces [cm]", constant=True)
        self.grid.addfield(
            "A", A, description="Radial grid annulus area [cm²]", constant=True)
