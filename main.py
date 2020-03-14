#!/usr/bin/python3

import os
import sys

from dolfin import *
import numpy as np

import solver as solver
import solver_emi as solver_emi
import plotter as plotter


def g_syn_hyper(g_syn_bar, a_syn, t):
    """ Stimulate axon, hyper activity """

    g_syn = Expression(
        "g_syn_bar*exp(-fmod(t,0.02)/a_syn)*\
                        (x[0] < 20.0e-6)*(x[1] < 0.5e-6)*(x[2] < 0.5e-6)",
        g_syn_bar=g_syn_bar,
        a_syn=a_syn,
        t=t,
        degree=4,
    )

    return g_syn


def g_syn(g_syn_bar, a_syn, t):
    """ Stimulate axon, normal activity """
    g_syn = Expression(
        "g_syn_bar*exp(-fmod(t,0.02)/a_syn)*\
                        (x[0] < 20.0e-6)*(x[1] < 0.5e-6)*(x[2] < 0.5e-6)*\
                        (t < 2.5e-3)",
        g_syn_bar=g_syn_bar,
        a_syn=a_syn,
        t=t,
        degree=4,
    )

    return g_syn


if __name__ == "__main__":

    # resolution factor of mesh
    resolution = 0

    # time variables (seconds)
    dt = 1.0e-4        # global time step (s)
    Tstop = 1.0e-1     # end time (s)
    n_steps_ode = 25   # number of steps for ODE solver

    # physical parameters
    C_M = 0.02         # capacitance (F)
    temperature = 300  # temperature (K)
    F = 96485          # Faraday's constant (C/mol)
    R = 8.314          # Gas constant (J/(K*mol))
    g_Na_bar = 1200    # Na max conductivity (S/m^2)
    g_K_bar = 360      # K max conductivity (S/m^2)

    g_Na_leak = Constant(0.1544) # Na leak membrane conductivity (S/(m^2))
    g_K_leak = Constant(0.3125)  # K leak membrane conductivity (S/(m^2))
    g_Cl_leak = Constant(0.2)    # Cl leak membrane conductivity (S/(m^2))

    # cotransporters
    g_KCC2 = 0.0034 # KCC2 cotransporter strength (A/m^2)
    g_NKCl = 0.023  # NKCl cotransporter strength (A/m^2)

    # pump
    I_max = 0.1804176 # max pump strength (A/m^2)
    m_K = 3.0         # threshold ECS K (mol/m^3)
    m_Na = 12.0       # threshold ICS Na (mol/m^3)

    a_syn = 0.002   # synaptic time constant (s)
    g_syn_bar = 40  # synaptic conductivity (A/m)

    D_Na = Constant(1.33e-9) # Na diffusion coefficient (m^2/s)
    D_K = Constant(1.96e-9)  # K diffusion coefficient (m^2/s)
    D_Cl = Constant(2.03e-9) # Cl diffusion coefficient (m^2/s)
    D_A = Constant(2.03e-9)  # Anion diffusion coefficient (m^2/s)

    z_A = Constant(-1.0)  # valence of anions

    # initial conditions
    phi_M_init = Constant(-0.0677379)   # membrane potential (V)
    V_rest = -0.065                     # resting membrane potential

    Na_i_init = Constant(18) # intracellular Na concentration (mol/m^3)
    K_i_init = Constant(80)  # intracellular K concentration (mol/m^3)
    Cl_i_init = Constant(7)  # intracellular Cl concentration (mol/m^3)
    A_i = Constant(91)       # intracellular anions (mol/m^3)

    Na_e_init = Constant(120) # extracellular Na concentration (mol/m^3)
    K_e_init = Constant(4)    # extracellular K concentration (mol/m^3)
    Cl_e_init = Constant(60)  # extracellular Cl concentration (mol/m^3)
    A_e = Constant(64)        # intracellular anions (mol/m^3)

    # EMI parameters
    sigma_i = 0.7276    # intracellular conductivity
    sigma_e = 1.079     # extracellular conductivity
    E_Na = 49.04e-3     # reversal potential Na (V)
    E_K = -77.44e-3     # reversal potential K (V)
    E_Cl = -55.54e-3    # reversal potential Cl (V)

    # set parameters
    params = {
        "dt": dt,
        "Tstop": Tstop,
        "temperature": temperature,
        "R": R,
        "F": F,
        "C_M": C_M,
        "phi_M_init": phi_M_init,
        "V_rest": V_rest,
        "g_K_bar": g_K_bar,
        "g_Na_bar": g_Na_bar,
        "I_max": I_max,
        "m_Na": m_Na,
        "m_K": m_K,
        "sigma_i": sigma_i,
        "sigma_e": sigma_e,
        "E_Na": E_Na,
        "E_K": E_K,
        "E_Cl": E_Cl,
        "g_KCC2": g_KCC2,
        "g_NKCl": g_NKCl,
        "g_Na_leak": g_Na_leak,
        "g_K_leak": g_K_leak,
        "g_Cl_leak": g_Cl_leak,
        "D_A": D_A,
        "A_i": A_i,
        "A_e": A_e,
        "z_A": z_A,
    }

    # create ions (Na conductivity is set below for each model)
    Na = {
        "Di": D_Na,
        "De": D_Na,
        "ki_init": Na_i_init,
        "ke_init": Na_e_init,
        "z": 1.0,
        "name": "Na",
    }
    K = {
        "Di": D_K,
        "De": D_K,
        "ki_init": K_i_init,
        "ke_init": K_e_init,
        "z": 1.0,
        "name": "K",
    }
    Cl = {
        "Di": D_Cl,
        "De": D_Cl,
        "ki_init": Cl_i_init,
        "ke_init": Cl_e_init,
        "z": -1.0,
        "name": "Cl",
    }

    # create ion list
    ion_list = [Na, K, Cl]

    #####################################################################
    # get mesh, subdomains, surfaces paths
    mesh_prefix = "meshes/two_neurons_3d/"
    mesh = mesh_prefix + "mesh_" + str(resolution) + ".xml"
    subdomains = mesh_prefix + "subdomains_" + str(resolution) + ".xml"
    surfaces = mesh_prefix + "surfaces_" + str(resolution) + ".xml"
    # generate mesh if it does not exist
    if not os.path.isfile(mesh):
        script = "make_mesh.py "  # script
        os.system("python " + script + " " + str(resolution))  # run script

    # Run KNP-EMI hyperactivity
    sys.stdout.write("\n--------------------------------")
    sys.stdout.write("\nRunning KNP-EMI hyperactivity")
    sys.stdout.write("\n--------------------------------\n")
    t_2a = Constant(0.0)  # time constant
    # file for results
    fname_knpemi_hyper = ("results/knpemi_hyper/res_" + str(resolution) + "/")  # filename for results
    # set ion channel conductivity
    ion_list[0]["g_k"] = g_Na_leak + g_syn_hyper(g_syn_bar, a_syn, t_2a)  # Na
    ion_list[1]["g_k"] = g_K_leak  # K
    ion_list[2]["g_k"] = g_Cl_leak  # Cl
    # solve system
    S_2a = solver.Solver(ion_list, t_2a, **params) # create solver
    S_2a.setup_domain(mesh, subdomains, surfaces)  # setup domains
    S_2a.solve_system_HH(n_steps_ode, filename=fname_knpemi_hyper)  # solve

    # Run EMI hyperactivity
    sys.stdout.write("\n--------------------------------")
    sys.stdout.write("\nRunning EMI hyperactivity")
    sys.stdout.write("\n--------------------------------\n")
    t_2b = Constant(0.0)  # time constant
    fname_emi_hyper = ("results/emi_hyper/res_" + str(resolution) + "/")  # filename for results
    # set synaptic current
    params["g_ch_syn"] = g_syn_hyper(g_syn_bar, a_syn, t_2b)
    # solve system
    S_2b = solver_emi.Solver(t_2b, **params)       # create solver
    S_2b.setup_domain(mesh, subdomains, surfaces)  # setup domains
    S_2b.solve_system_HH(n_steps_ode, filename=fname_emi_hyper)  # solve

    # Run KNP-EMI normal activity
    sys.stdout.write("\n--------------------------------")
    sys.stdout.write("\nRunning KNP-EMI normal activity")
    sys.stdout.write("\n--------------------------------\n")
    t_1a = Constant(0.0)  # time constant
    # file for results
    fname_knpemi = "results/knpemi/res_" + str(resolution) + "/"  # filename for results
    # set ion channel conductivity
    ion_list[0]["g_k"] = g_Na_leak + g_syn(g_syn_bar, a_syn, t_1a)  # Na
    ion_list[1]["g_k"] = g_K_leak  # K
    ion_list[2]["g_k"] = g_Cl_leak  # Cl
    # solve system
    S_1a = solver.Solver(ion_list, t_1a, **params) # create solver
    S_1a.setup_domain(mesh, subdomains, surfaces)  # setup domains
    S_1a.solve_system_HH(n_steps_ode, filename=fname_knpemi)  # solve

    # Run EMI normal activity
    sys.stdout.write("\n--------------------------------")
    sys.stdout.write("\nRunning EMI normal activity")
    sys.stdout.write("\n--------------------------------\n")
    t_1b = Constant(0.0)  # time constant
    fname_emi = "results/emi/res_" + str(resolution) + "/"  # filename for results
    # set synaptic current
    params["g_ch_syn"] = g_syn(g_syn_bar, a_syn, t_1b)
    # solve system
    S_1b = solver_emi.Solver(t_1b, **params)      # create solver
    S_1b.setup_domain(mesh, subdomains, surfaces) # setup domains
    S_1b.solve_system_HH(n_steps_ode, filename=fname_emi)  # solve

    # files containing solutions
    f1 = fname_knpemi + "results.h5"
    f2 = fname_emi + "results.h5"
    f3 = fname_knpemi_hyper + "results.h5"
    f4 = fname_emi_hyper + "results.h5"

    # create plotter and generate plots
    sys.stdout.write("\n--------------------------------")
    sys.stdout.write("\nCreating plots")
    sys.stdout.write("\n--------------------------------\n")
    P = plotter.Plotter(resolution, Tstop, dt * 10, f1, f2, f3, f4)
    P.make_figures()
