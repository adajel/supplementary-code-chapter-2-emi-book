import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from fenics import *

# set font & text parameters
font = {"family": "serif", "weight": "bold", "size": 13}

plt.rc("font", **font)
plt.rc("text", usetex=True)
mpl.rcParams["image.cmap"] = "jet"


class Plotter:
    def __init__(self, res_2D, T, dt, f, f_emi, f_hyper, f_emi_hyper):
        # plotting parameters
        self.res_2D = res_2D # mesh resolution
        self.T = T           # end time
        self.dt = 1.0e-3

        # create time series
        self.time = 1.0e3 * np.arange(dt, T, dt)

        # files containing solutions
        self.f = f                      # KNP-EMI normal activity
        self.f_hyper = f_hyper          # KNP-EMI hyperactivity
        self.f_emi = f_emi              # EMI normal activity
        self.f_emi_hyper = f_emi_hyper  # EMI normal activity

        return

    def get_plottable_ECS_function(self, h5_fname, n, i, scale=1.0):
        """ get plottable function of extracellular concentration or potential """

        mesh = Mesh()
        subdomains = MeshFunction("size_t", mesh, 2)
        surfaces = MeshFunction("size_t", mesh, 1)
        hdf5 = HDF5File(MPI.comm_world, h5_fname, "r")
        hdf5.read(mesh, "/mesh", False)
        mesh.coordinates()[:] *= 1e6
        hdf5.read(subdomains, "/subdomains")
        hdf5.read(surfaces, "/surfaces")

        exterior_mesh = SubMesh(mesh, subdomains, 0)
        P1 = FiniteElement("CG", triangle, 1)
        R = FiniteElement("R", triangle, 0)

        # EMI
        if i is None:
            We = FunctionSpace(exterior_mesh, MixedElement([P1] + [R]))
        # KNP-EMI
        else:
            We = FunctionSpace(exterior_mesh, MixedElement(4 * [P1] + [R]))

        Ve = FunctionSpace(exterior_mesh, P1)
        ue = Function(We)
        fe = Function(Ve)

        # EMI
        if i is None:
            hdf5.read(ue, "/exterior_solution/vector_" + str(n))
            assign(fe, ue.sub(0))
        # KNP-EMI
        else:
            hdf5.read(ue, "/exterior_solution/vector_" + str(n))
            assign(fe, ue.sub(i))

        # scale (e.g. from V to mV)
        fe.vector()[:] = scale * fe.vector().get_local()

        return fe

    def get_time_series_ECS(self, dt, T, fname, x, y, z, EMI=False):
        """ Return list of values in given point (x, y, z) over time """
        # read data file
        hdf5file = HDF5File(MPI.comm_world, fname, "r")

        # create mesh
        mesh = Mesh()
        subdomains = MeshFunction("size_t", mesh, 2)
        surfaces = MeshFunction("size_t", mesh, 1)
        hdf5file.read(mesh, "/mesh", False)
        mesh.coordinates()[:] *= 1e6
        hdf5file.read(subdomains, "/subdomains")
        hdf5file.read(surfaces, "/surfaces")
        exterior_mesh = SubMesh(mesh, subdomains, 0)

        # define function spaces
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        We = FunctionSpace(exterior_mesh, MixedElement(4 * [P1]))
        Ve = FunctionSpace(exterior_mesh, P1)

        # define functions
        ue = Function(We)
        f_phi_e = Function(Ve)
        f_Na_e = Function(Ve)
        f_K_e = Function(Ve)

        # list of values at point over time
        Na_e = np.empty(int(T / dt) - 1)
        K_e = np.empty(int(T / dt) - 1)
        phi_e = np.empty(int(T / dt) - 1)

        for n in range(1, int(T / dt)):
            if EMI:
                # read file and append membrane potential
                hdf5file.read(f_phi_e, "/exterior_solution/vector_" + str(n))
                # check if 2D or 3D
                phi_e[n - 1] = 1.0e3 * f_phi_e(x, y, z)  # 3D
            else:
                # read file
                hdf5file.read(ue, "/exterior_solution/vector_" + str(n))

                assign(f_Na_e, ue.sub(0))   # ECS Na concentrations
                assign(f_K_e, ue.sub(1))    # ECS K concentrations
                assign(f_phi_e, ue.sub(3))  # ECS potential

                phi_e[n - 1] = 1.0e3 * f_phi_e(x, y, z)
                Na_e[n - 1] = f_Na_e(x, y, z)
                K_e[n - 1] = f_K_e(x, y, z)

        return Na_e, K_e, phi_e

    def get_time_series_gamma(self, dt, T, fname, x, y, z):
        """ Return list of values in given point (x, y, z) over time """
        # read data file
        hdf5file = HDF5File(MPI.comm_world, fname, "r")

        # membrane potential
        gamma_mesh = Mesh()
        hdf5file.read(gamma_mesh, "/gamma_mesh", False)
        P1 = FiniteElement("P", gamma_mesh.ufl_cell(), 1)
        Vg = FunctionSpace(gamma_mesh, P1)
        gamma_mesh.coordinates()[:] *= 1e6

        f_phi_M = Function(Vg)
        phi_M = []

        for n in range(1, int(T / dt)):
            # read file
            hdf5file.read(f_phi_M, "/membrane_potential/vector_" + str(n))
            # membrane potential
            phi_M.append(1.0e3 * f_phi_M(x, y, z))

        return phi_M

    def make_figures(self):
        """ Create plots of potentials and ion concentrations """
        # set time parameters
        dt = self.dt
        T = self.T
        time = self.time

        # point at membrane
        x_M = 100; y_M = 0.5; z_M = 0.5
        # point in ECS (1 um above axon)
        x_E = 100; y_E = 0.5 + 0.05; z_E = 0.5

        # get time series - membrane potential
        phi_M = self.get_time_series_gamma(dt, T, self.f, x_M, y_M, z_M)
        phi_M_emi = self.get_time_series_gamma(dt, T, self.f_emi, x_M, y_M, z_M)
        # get time series - ECS ion concentrations
        Na, K, phi_E = self.get_time_series_ECS(dt, T, self.f, x_E, y_E, z_E)
        _, _, phi_E_emi = self.get_time_series_ECS(
            dt, T, self.f_emi, x_E, y_E, z_E, EMI=True
        )

        # get time series - membrane potential
        phi_M_hyper = self.get_time_series_gamma(dt, T, self.f_hyper, x_M, y_M, z_M)
        phi_M_emi_hyper = self.get_time_series_gamma(
            dt, T, self.f_emi_hyper, x_M, y_M, z_M
        )
        # get time series - ion concentrations
        Na_hyper, K_hyper, phi_E_hyper = self.get_time_series_ECS(
            dt, T, self.f_hyper, x_E, y_E, z_E
        )
        _, _, phi_E_emi_hyper = self.get_time_series_ECS(
            dt, T, self.f_emi_hyper, x_E, y_E, z_E, EMI=True
        )

        # create figure for potential
        fig = plt.figure(figsize=(12 / 1.5, 25 / 3.0))
        gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.3, hspace=0.3)

        # membrane potential during normal activity
        ax1 = fig.add_subplot(gs[0, 0])
        plt.title(r"Membrane potential")
        plt.ylabel(r"$\phi_M$ (mV)")
        plt.xlabel(r"time (ms)")
        plt.yticks([-80, -60, -40, -20, 0, 20])
        plt.ylim(-100, 40)
        plt.plot(time, phi_M, linewidth=3, label="KNP-EMI")
        plt.plot(time, phi_M_emi, "--", linewidth=3, label="EMI")
        plt.legend()
        # membrane potential during hyperactivity
        ax2 = fig.add_subplot(gs[1, 0])
        plt.title(r"Membrane potential")
        plt.ylabel(r"$\phi_M$ (mV)")
        plt.xlabel(r"time (ms)")
        plt.yticks([-80, -60, -40, -20, 0, 20])
        plt.ylim(-100, 40)
        plt.plot(time, phi_M_hyper, linewidth=3, label="KNP-EMI")
        plt.plot(time, phi_M_emi_hyper, "--", linewidth=3, label="EMI")
        # ECS potential during normal activity
        ax3 = fig.add_subplot(gs[0, 1])
        plt.title(r"ECS potential")
        plt.ylabel(r"$\phi_M$ (mV)")
        plt.xlabel(r"time (ms)")
        plt.yticks([-1.0, -0.50, 0, 0.50, 1.0])
        plt.ylim(-1.15, 1.15)
        plt.plot(time, phi_E, linewidth=3, label="KNP-EMI")
        plt.plot(time, phi_E_emi, "--", linewidth=3, label="EMI")
        # ECS potential during hyperactivity
        ax4 = fig.add_subplot(gs[1, 1])
        plt.title(r"ECS potential")
        plt.ylabel(r"$\phi_M$ (mV)")
        plt.xlabel(r"time (ms)")
        plt.yticks([-1.0, -0.50, 0, 0.50, 1.0])
        plt.ylim(-1.15, 1.15)
        plt.plot(time, phi_E, linewidth=3, label="KNP-EMI")
        plt.plot(time, phi_E_emi, "--", linewidth=3, label="EMI")
        # add numbering for the subplots (A, B, C etc)
        letters = [r"\textbf{A}", r"\textbf{C}", r"\textbf{B}", r"\textbf{D}"]
        for n, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.1, 1.1, letters[n], transform=ax.transAxes)
        # save figure
        plt.savefig("potentials.svg", format="svg")

        # create figure for ECS concentrations
        fig = plt.figure(figsize=(12 / 1.5, 25 / 3.0))
        gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.93, wspace=0.3, hspace=0.3)

        # ECS Na concentration during normal activity
        ax1 = fig.add_subplot(gs[0, 0])
        plt.title(r"ECS Na$^+$ concentration")
        plt.ylabel(r"$c^{Na}_e$(mM)")
        plt.xlabel(r"time (ms)")
        plt.yticks([119.4, 119.6, 119.8, 120])
        plt.ylim([119.3, 120.1])
        plt.plot(time, Na, linewidth=3)
        # ECS Na concentration during hyperactivity
        ax2 = fig.add_subplot(gs[1, 0])
        plt.title(r"ECS Na$^+$ concentration")
        plt.ylabel(r"$c^{Na}_e$(mM)")
        plt.xlabel(r"time (ms)")
        plt.yticks([119.4, 119.6, 119.8, 120])
        plt.ylim([119.3, 120.1])
        plt.plot(time, Na_hyper, linewidth=3)
        # ECS K concentration during normal activity
        ax3 = fig.add_subplot(gs[0, 1])
        plt.title(r"ECS K$^+$ concentration")
        plt.ylabel(r"$c^{K}_e$(mM)")
        plt.xlabel(r"time (ms)")
        plt.yticks([4.0, 4.2, 4.4, 4.6, 4.8])
        plt.ylim([3.96, 4.84])
        plt.plot(time, K, linewidth=3)
        # ECS K concentration during hyperactivity
        ax4 = fig.add_subplot(gs[1, 1])
        plt.title(r"ECS K$^+$ concentration")
        plt.ylabel(r"$c^{K}_e$(mM)")
        plt.xlabel(r"time (ms)")
        plt.yticks([4.0, 4.2, 4.4, 4.6, 4.8])
        plt.ylim([3.96, 4.84])
        plt.plot(time, K_hyper, linewidth=3)

        # add numbering for the subplots (A, B, C etc)
        letters = [r"\textbf{A}", r"\textbf{C}", r"\textbf{B}", r"\textbf{D}"]
        for n, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.text(-0.1, 1.1, letters[n], transform=ax.transAxes)
        # save figure
        plt.savefig("concentrations.svg", format="svg")

        return
