import numpy as np


def get_rhos_simple(a, rhos, smin, smax):
    """
    Function to compute the bulk density of the reconstructed particle sizes.
    This simple model assumes the bulk density to be constant within the
    two original particle size bins.

    Parameters
    ----------
    a : array-like
        Particle size for which bulk densities should be computed
    rhos : array-like
        Bulk densities in the two population model
    smin : array-like
        Minimum particle sizes
    smax : array-like
        Maximum particle sizes

    Returns
    -------
    rhos_recon : array-like
        Reconstructed particle bulk densities
    """
    sint = np.sqrt(smin*smax)
    rhos_recon = np.ones_like(a[:, None, :]) * rhos[..., 1, None]
    rhos_recon = np.where(
        a[:, None, :] < sint[..., None],
        rhos[..., 0, None],
        rhos_recon
    )
    return rhos_recon


def get_q(Sigma, smin, smax):
    """
    Function computes the power law exponent of the size distribution
    n(a) da = a^q da

    Parameters
    ----------
    Sigma : array-like
        Dust surface densities
    smin : array-like
        Minimum particle sizes
    smax : array-like
        Maximum particle sizes

    Returns
    -------
    q : array-like
        Size distribution exponent
    """
    sint = np.sqrt(smin*smax)
    return -(np.log(Sigma[..., 1]/Sigma[..., 0]) / np.log(smax/sint) - 4.)


def get_size_distribution(sigma_d, a_max, q=3.5, na=10, agrid_min=None, agrid_max=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density
    where the power-law can be a single float or different in each radial bin.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array, shape (nr,) or (nr, nphi)

    a_max : array
        maximum particle size array, shape (nr,) or (nr, nphi)

    Keywords:
    ---------

    q : float | array
        particle size index, n(a) propto a**-q
        scalar, shape (nr,), or shape (nr, naz) matching sigma_d

    na : int
        number of particle size bins

    agrid_min : float
        minimum particle size

    agrid_max : float
        maximum particle size of the grid

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution, shape (nr, na) for 1-D input or
        (nr, nphi, na) for 2-D input, units of g/cm^2 integrated over bins.
    """

    sigma_d = np.asarray(sigma_d)
    a_max = np.asarray(a_max)
    input_ndim = sigma_d.ndim
    nd_none = (None,) * input_ndim
    size_idx = nd_none + (slice(None),)
    q_arr = np.array(q)[..., None]

    if agrid_min is None:
        agrid_min = a_max.min()

    if agrid_max is None:
        agrid_max = 2 * a_max.max()

    a_i = np.logspace(np.log10(agrid_min), np.log10(agrid_max), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    # our cell integral goes always from the lower interface up to either the upper interface or to a_max
    a_left = a_i[:-1][size_idx]
    a_right = np.where(
        a_i[1:][size_idx] < a_max[..., None],
        a_i[1:][size_idx],
        np.maximum(a_max[..., None], a_i[:-1][size_idx]))

    sig_da = sigma_d[..., None] * np.where(
        q_arr == 4,
        np.log(a_right / a_left) / np.log(a_max / agrid_min)[..., None],
        (a_right**(4 - q_arr) - a_left**(4 - q_arr)) / (a_max[..., None]**(4 - q_arr) - agrid_min**(4 - q_arr)))

    return a, a_i, sig_da


def sim_size_distribution(sim, comp_name=None, agrid_min=None, agrid_max=None, Nm=None):
    """
    Computes the size distribution for a given component in the simulation.

    Arguments:
    ----------

    sim : Simulation
        The simulation object containing the component.

    Optional arguments:

    component_name : str
        The name of the component for which to compute the size distribution.
            If None, the size distribution is computed for the total dust surface density and qrec.

    agrid_min : float
        Minimum particle size for the size distribution grid.
        Default is the minimum particle size in the simulation.

    agrid_max : float
        Maximum particle size for the size distribution grid.
        Default is 1.5 times the maximum particle size in the simulation.

    Nm : int
        Number of size bins for the size distribution grid.
        Default is 7 mass bins per mass decade, which corresponds to 7*log10(mgrid_max/mgrid_min) size bins.

    Returns:
    --------

    a : array
        Particle size grid (centers).

    a_i : array
        Particle size grid (interfaces).

    sig_da : array
        Particle size distribution of size (len(sigma_d), na), units of g/cm^2.
    """
    # set default grid limits if not provided
    if agrid_min is None:
        agrid_min = min(sim.dust.s.min)

    if agrid_max is None:
        agrid_max = max(sim.dust.s.max)*1.5

    if comp_name is None:
        sigma_d = sim.dust.Sigma.sum(axis=-1)
        q = np.abs(sim.dust.qrec)
    else:
        comp = sim.components.__dict__[comp_name]
        sigma_d = comp.dust.Sigma.sum(axis=-1)
        q = np.abs((np.log(comp.dust.Sigma[:, 1]/comp.dust.Sigma[:, 0]))/np.log(
            sim.dust.s.max/np.sqrt(sim.dust.s.min*sim.dust.s.max)) - 4.)

    # Dustpy like size distribution grid is logarithmic in mass -> Tripodpy assimes rhos = constant -> logarithmic in size
    if Nm is None:
        Nmbpd = 7  # number of mass bins per decade
        logmmin = np.log10(4./3.*np.pi*agrid_min**3*sim.dust.rhos.min())
        logmmax = np.log10(4./3.*np.pi*agrid_max**3*sim.dust.rhos.max())
        decades = np.ceil(logmmax - logmmin)
        Nm = int(decades * Nmbpd) + 1

    return get_size_distribution(sigma_d, sim.dust.s.max, q=q, na=Nm, agrid_min=agrid_min, agrid_max=agrid_max)
