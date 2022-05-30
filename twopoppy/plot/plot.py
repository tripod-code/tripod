import twopoppy.constants as c
from twopoppy.plot.read_data import _readdata_tpp as _readdata

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def panel(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max() / c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max - (levels_num - 1), sd_max, levels_num)

    width = 3.5
    fig = plt.figure(figsize=(3. * width, 2. * width / 1.618), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.r[it, ...] / c.au,
                          data.mi[it, ir, ...],
                          np.log10(data.sigmaDusti[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    if show_St1:
        ax00.contour(data.r[it, ...] / c.au,
                     data.mi[it, ir, ...],
                     data.Sti[it, ...].T,
                     levels=[1.],
                     colors="white",
                     linewidths=2
                     )
    if show_limits:
        ax00.contour(data.r[it, ...] / c.au,
                     data.mi[it, ir, ...],
                     (data.Sti - data.StDr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C2",
                     linewidths=1
                     )
        ax00.contour(data.r[it, ...] / c.au,
                     data.mi[it, ir, ...],
                     (data.Sti - data.StFr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C0",
                     linewidths=1
                     )

    ax00.axhline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
    ax00.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle mass [g]")

    ax01.loglog(data.mi[it, ir, ...], data.sigmaDusti[it, ir, :], c="C3")
    ax01.set_xlim(data.mi[it, ir, 0], data.mi[it, ir, -1])
    ax01.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
    ax01.set_xlabel("Particle mass [g]")
    ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    if data.Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t / c.year, data.Mgas / c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t / c.year, data.Mdust /
                    c.M_sun, c="C1", label="Dust")
        ax02.axvline(data.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1] / c.year, data.t[-1] / c.year)
        ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel("Mass [$M_\odot$]")

    ax10.loglog(data.r[it, ...] / c.au, data.sigmaDusti[it, :, im], c="C3")
    ax10.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
    ax10.set_ylim(10. ** (sd_max - 6.), 10. ** sd_max)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    ax11.loglog(data.r[it, ...] / c.au, data.SigmaGas[it, ...], label="Gas")
    ax11.loglog(data.r[it, ...] / c.au,
                data.SigmaDusti[it, ...].sum(-1), label="Dust")
    ax11.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
    ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel("$\Sigma$ [g/cm²]")
    ax11.legend()
    ax11r.loglog(data.r[it, ...] / c.au, data.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    fig.text(0.99, 0.01, "TwoPopPy v" + __version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()


def ipanel(data, filename="data", extension="hdf5", im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple interactive plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``twopoppy.Simulation`` or string
        Either instance of ``twopoppy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from twopoppy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt - 1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nmi[it, ...] - 1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...] - 1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDusti.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max() / c.M_sun)) + 1
    levels_num = 7
    levels = np.linspace(sd_max - (levels_num - 1), sd_max, levels_num)

    width = 3.5
    fig = plt.figure(figsize=(3. * width, 2. * width / 1.618), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.r[it, ...] / c.au,
                          data.mi[it, ir, ...],
                          np.log10(data.sigmaDusti[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    plt00Collections = plt00.collections[:]
    if show_St1:
        plt00St = ax00.contour(data.r[it, ...] / c.au,
                               data.mi[it, ir, ...],
                               data.Sti[it, ...].T,
                               levels=[1.],
                               colors="white",
                               linewidths=2
                               )
        plt00StCollections = plt00St.collections[:]
    if show_limits:
        plt00Dr = ax00.contour(data.r[it, ...] / c.au,
                               data.mi[it, ir, ...],
                               (data.Sti - data.StDr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C2",
                               linewidths=1
                               )
        plt00DrCollections = plt00Dr.collections[:]
        plt00Fr = ax00.contour(data.r[it, ...] / c.au,
                               data.mi[it, ir, ...],
                               (data.Sti - data.StFr[..., None])[it, ...].T,
                               levels=[0.],
                               colors="C0",
                               linewidths=1
                               )
        plt00FrCollections = plt00Fr.collections[:]
    plt00hl = ax00.axhline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
    plt00vl = ax00.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle mass [g]")

    plt01 = ax01.loglog(data.mi[it, ir, ...], data.sigmaDusti[it, ir, :], c="C3")
    plt01vl = ax01.axvline(data.mi[it, ir, im], color="#AAAAAA", lw=1, ls="--")
    ax01.set_xlim(data.mi[it, ir, 0], data.mi[it, ir, -1])
    ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
    ylim0 = ylim1 - 6.
    ax01.set_ylim(10. ** ylim0, 10. ** ylim1)
    ax01.set_xlabel("Particle mass [g]")
    ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    if data.Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t / c.year, data.Mgas / c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t / c.year, data.Mdust /
                    c.M_sun, c="C1", label="Dust")
        plt02vl = ax02.axvline(data.t[it] / c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1] / c.year, data.t[-1] / c.year)
        ax02.set_ylim(10. ** (Mmax - 6.), 10. ** Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel("Mass [$M_\odot$]")

    plt10 = ax10.loglog(data.r[it, ...] / c.au,
                        data.sigmaDusti[it, :, im], c="C3")
    plt10vl = ax10.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
    ax10.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
    ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
    ylim0 = ylim1 - 6.
    ax10.set_ylim(10. ** ylim0, 10. ** ylim1)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    plt11g = ax11.loglog(data.r[it, ...] / c.au,
                         data.SigmaGas[it, ...], label="Gas")
    plt11d = ax11.loglog(data.r[it, ...] / c.au,
                         data.SigmaDusti[it, ...].sum(-1), label="Dust")
    plt11vl = ax11.axvline(data.r[it, ir] / c.au, color="#AAAAAA", lw=1, ls="--")
    ax11.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
    ax11.set_ylim(10. ** (sg_max - 6), 10. ** sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel("$\Sigma$ [g/cm²]")
    ax11.legend()
    plt11d2g = ax11r.loglog(data.r[it, ...] / c.au,
                            data.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    width = ax02.get_position().x1 - ax02.get_position().x0
    fig._widgets = []

    if data.Nt > 2:
        axSliderTime = plt.axes([ax02.get_position().x0 + 0.15 * width,
                                 0.375,
                                 0.75 * width,
                                 0.02], facecolor="lightgoldenrodyellow")
        sliderTime = Slider(axSliderTime, "Time", 0, data.Nt -
                            1, valinit=it, valfmt="%i")
        axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it] / c.year))
        fig._widgets += [sliderTime]

    axSliderMass = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.25,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderMass = Slider(axSliderMass, "Mass", 0,
                        data.Nmi[it] - 1, valinit=im, valfmt="%i")
    axSliderMass.set_title("m = {:9.3e} g".format(data.mi[it, ir, im]))
    fig._widgets += [sliderMass]

    axSliderDist = plt.axes([ax02.get_position().x0 + 0.15 * width,
                             0.125,
                             0.75 * width,
                             0.02], facecolor="lightgoldenrodyellow")
    sliderDist = Slider(axSliderDist, "Distance", 0,
                        data.Nr[it] - 1, valinit=ir, valfmt="%i")
    axSliderDist.set_title("r = {:9.3e} AU".format(data.r[it, ir] / c.au))
    fig._widgets += [sliderDist]

    def update(val):

        it = 0
        if data.Nt > 2:
            it = int(np.floor(sliderTime.val))
            axSliderTime.set_title("t = {:9.3e} yr".format(data.t[it] / c.year))
        im = int(np.floor(sliderMass.val))
        ir = int(np.floor(sliderDist.val))
        axSliderMass.set_title("m = {:9.3e} g".format(data.mi[it, ir, im]))
        axSliderDist.set_title("r = {:9.3e} AU".format(data.r[it, ir] / c.au))

        for row in plt00Collections:
            ax00.collections.remove(row)
            plt00Collections.remove(row)
        plt00 = ax00.contourf(data.r[it, ...] / c.au,
                              data.mi[it, ir, ...],
                              np.log10(data.sigmaDusti[it, ...].T),
                              levels=np.linspace(sd_max - 6, sd_max, 7),
                              cmap="magma",
                              extend="both"
                              )
        for row in plt00.collections:
            plt00Collections.append(row)
        if show_St1:
            for row in plt00StCollections:
                ax00.collections.remove(row)
                plt00StCollections.remove(row)
            plt00St = ax00.contour(data.r[it, ...] / c.au,
                                   data.mi[it, ir, ...],
                                   data.Sti[it, ...].T,
                                   levels=[1.],
                                   colors="white",
                                   linewidths=2
                                   )
            for row in plt00St.collections:
                plt00StCollections.append(row)
        if show_limits:
            for row in plt00DrCollections:
                ax00.collections.remove(row)
                plt00DrCollections.remove(row)
            plt00Dr = ax00.contour(data.r[it, ...] / c.au,
                                   data.mi[it, ir, ...],
                                   (data.Sti - data.StDr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C2",
                                   linewidths=1
                                   )
            for row in plt00Dr.collections:
                plt00DrCollections.append(row)
            for row in plt00FrCollections:
                ax00.collections.remove(row)
                plt00FrCollections.remove(row)
            plt00Fr = ax00.contour(data.r[it, ...] / c.au,
                                   data.mi[it, ir, ...],
                                   (data.Sti - data.StFr[..., None])[it, ...].T,
                                   levels=[0.],
                                   colors="C0",
                                   linewidths=1
                                   )
            for row in plt00Fr.collections:
                plt00FrCollections.append(row)
        plt00vl.set_xdata([data.r[it, ir] / c.au, data.r[it, ir] / c.au])
        plt00hl.set_ydata([data.mi[it, ir, im], data.mi[it, ir, im]])

        plt01[0].set_xdata(data.mi[it, ir, ...])
        plt01[0].set_ydata(data.sigmaDusti[it, ir, :])
        ax01.set_xlim(data.mi[it, ir, 0], data.mi[it, ir, -1])
        ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, ir, :])))
        ylim0 = ylim1 - 6.
        ax01.set_ylim(10. ** ylim0, 10. ** ylim1)
        plt01vl.set_xdata([data.mi[it, ir, im], data.mi[it, ir, im]])
        plt01vl.set_ydata([0., 1.e100])

        if data.Nt > 2:
            plt02vl.set_xdata([data.t[it] / c.year, data.t[it] / c.year])
        plt10vl.set_xdata([data.r[it, ir] / c.au, data.r[it, ir] / c.au])
        plt11vl.set_xdata([data.r[it, ir] / c.au, data.r[it, ir] / c.au])

        plt10[0].set_xdata(data.r[it, ...] / c.au)
        plt10[0].set_ydata(data.sigmaDusti[it, :, im])
        ax10.set_xlim(data.r[it, 0] / c.au, data.r[it, -1] / c.au)
        ylim1 = np.ceil(np.log10(np.max(data.sigmaDusti[it, :, im])))
        ylim0 = ylim1 - 6.
        ax10.set_ylim(10. ** ylim0, 10. ** ylim1)
        plt10vl.set_xdata([data.r[it, ir] / c.au, data.r[it, ir] / c.au])
        plt10vl.set_ydata([0., 1.e100])

        plt11g[0].set_xdata(data.r[it, ...] / c.au)
        plt11g[0].set_ydata(data.SigmaGas[it, ...])
        plt11d[0].set_xdata(data.r[it, ...] / c.au)
        plt11d[0].set_ydata(data.SigmaDusti[it, ...].sum(-1))
        plt11vl.set_xdata([data.r[it, ir] / c.au, data.r[it, ir]])
        plt11vl.set_ydata([0., 1.e100])
        plt11d2g[0].set_xdata(data.r[it, ...] / c.au)
        plt11d2g[0].set_ydata(data.eps[it, ...])

    if data.Nt > 2:
        sliderTime.on_changed(update)
    sliderMass.on_changed(update)
    sliderDist.on_changed(update)

    fig.text(0.99, 0.01, "Twopoppy v" + __version__, horizontalalignment="right",
             verticalalignment="bottom")

    plt.show()
