
"""Module containing standard functions for the composition"""
import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from tripod.std import dust_f
import dustpy as dp
import numpy as np
import scipy.sparse as sp


def prepare(sim):
    """Function prepares implicit dust integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    Nm_s = int(sim.grid._Nm_short)
    Nr = int(sim.grid.Nr)

    # Copy values to state vector Y
    for name, comp in sim.gas.components.__dict__.items():
        if(name.startswith("_") or not comp.includedust):
            continue

        comp._Y[:Nr] = comp.Sigma.ravel()
        comp._Y[Nr:] = comp.dust.Sigma_dust.ravel()
    
def finalize(sim):
    """Function finalizes implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""
    Nm_s = int(sim.grid._Nm_short)
    Nr = int(sim.grid.Nr)

    for name, comp in sim.gas.components.__dict__.items():
        if(name.startswith("_") or not comp.includedust):
            continue

        comp.Sigma = comp._Y[:Nr].reshape(comp.Sigma.shape)
        comp.dust.Sigma_dust = comp._Y[Nr:].reshape(comp.dust.Sigma_dust.shape)



def Y_jacobian(sim, x, dx=None, *args, **kwargs):
    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx

    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)

    # Getting the Jacobian of Sigma
    J_Sigma = sim.dust.Sigma.jacobian(x, dx=dt)

    # Getting the Jacobian of Gas
    J_Gas =  dp.std.gas.jacobian(sim,x, dx= dt)

    # Stitching the Jacobians together
    J = J_Sigma.copy()
    J.data = np.hstack((J_Gas.data, J_Sigma.data))
    J.indices = np.hstack((J_Gas.indices, J_Sigma.indices + J_Gas.shape[0]))
    J.indptr = np.hstack((J_Gas.indptr, J_Sigma.indptr[1:] + len(J_Gas.data)))
    Ntot = J_Gas.shape[0] + J_Sigma.shape[0]
    J._shape = (Ntot, Ntot)

    return J

def _f_impl_1_direct_compo(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """Implicit 1st-order integration scheme with direct matrix inversion
    Parameters
    ----------
    x0 : Intvar
        Integration variable at beginning of scheme
    Y0 : Field
        Variable to be integrated at the beginning of scheme
    dx : IntVar
        Stepsize of integration variable
    jac : Field, optional, defaul : None
        Current Jacobian. Will be calculated, if not set
    args : additional positional arguments
    kwargs : additional keyworda arguments
    Returns
    -------
    dY : Field
        Delta of variable to be integrated
    Butcher tableau
    ---------------
     1 | 1
    ---|---
       | 1
    """
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = np.array(Y0.ravel())


    # Add external/explicit source terms to right-hand side
    name = kwargs.get("name")
    comp = Y0._owner.gas.components.__dict__.get(name)

    r = Y0._owner.grid.r
    ri = Y0._owner.grid.ri
    area = Y0._owner.grid.A
    Nr = int(Y0._owner.grid.Nr)
    Nm_s = int(Y0._owner.grid._Nm_short)

    # Set the right-hand side to 0 for the dust to be handeled like the global dust
    if Y0._owner.dust.boundary.inner.condition.startswith("const"):
        rhs[Nr:Nr+Nm_s] = 0.

    if Y0._owner.dust.boundary.inner.condition.startswith("const"):
        rhs[-Nm_s:] = 0.

    S = np.hstack((comp.S.ext.ravel(), comp.dust.Sext_dust.ravel()))

    # Right hand side
    rhs[...] += dx * S 

    N = jac.shape[0]
    eye = sp.identity(N, format="csc")

    A = eye - dx * jac

    A_LU = sp.linalg.splu(A,
                          permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
    Y1_ravel = A_LU.solve(rhs)

    Y1 = Y1_ravel.reshape(Y0.shape)

    return Y1 - Y0