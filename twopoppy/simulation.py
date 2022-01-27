from xml.dom.minidom import Attr
import dustpy as dp
import dustpy.constants as c
import numpy as np
from simframe import Instruction
from simframe import Integrator
from simframe.frame import Field


class Simulation(dp.Simulation):

    # Exclude the following functions from the from DustPy inherited object
    _excludefromparent = [
        "checkmassconservation",
        "setdustintegrator"
    ]

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Todo: Deleting not needed entries from ini object

        # Deleting Fields that are not needed
        del(self.grid.m)
        del(self.grid.Nm)

    # Note: the next two functions are to hide methods from DustPy that are not used in TwoPopPy
    # I have to check if there is a cleaner way of doing this.
    def __getattribute__(self, __name):
        if __name in super(dp.Simulation, self).__getattribute__("_excludefromparent"):
            raise AttributeError(__name)
        return super(dp.Simulation, self).__getattribute__(__name)

    def __dir__(self):
        return sorted((set(dir(self.__class__)) | set(self.__dict__.keys())) - set(self._excludefromparent))

    def initialize(self):
        '''Function initializes the simulation frame.

        Function sets all fields that are None with a standard value.
        If the grids are not set, it will call ``Simulation.makegrids()`` first.'''

        if not isinstance(self.grid.Nr, Field):
            self.makegrids()

        # Set integration variable
        if self.t is None:
            self.addintegrationvariable("t", 0., description="Time [s]")
            self.t.cfl = 0.1

            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific time step function
            self.t.updater = dp.std.sim.dt

            self.t.snapshots = np.logspace(3., 5., num=21, base=10.) * c.year

        # Initialize groups
        self._initializestar()
        self._initializegrid()
        self._initializegas()
        self._initializedust()

        # Set integrator
        if self.integrator is None:
            # Todo: Add instructions for dust quantities
            instructions = [
                Instruction(dp.std.gas.impl_1_direct,
                            self.gas.Sigma,
                            controller={"rhs": self.gas._rhs
                                        },
                            description="Gas: implicit 1st-order direct solver"
                            ),
            ]
            self.integrator = Integrator(
                self.t, description="Default integrator")
            self.integrator.instructions = instructions

        # Set writer
        if self.writer is None:
            self.writer = dp.utils.hdf5writer()

    def run(self):
        """This functions runs the simulation."""
        # Print welcome message
        if self.verbosity > 0:
            msg = ""
            msg += "\TwoPopPy v{}".format(self.__version__)
            msg += "\n"
            print(msg)
        # Actually run the simulation
        super(dp.Simulation, self).run()

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

    def _initializedust(self):
        '''Function to initialize dust quantities'''
        # Todo: write this function
        pass
