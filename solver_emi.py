#!/usr/bin/python3

from dolfin import *
import numpy as np
from petsc4py import PETSc
import sys

# turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)


class Solver:
    def __init__(self, t, **params):
        """ Initialize solver """

        self.params = params    # set parameters
        self.dt = params["dt"]  # set global time step (s)
        self.t = t              # set time constant (s)

        return

    def setup_domain(self, mesh_path, subdomains_path, surfaces_path):
        """ Setup mortar domain """

        # get and set mesh
        mesh = Mesh(mesh_path)
        self.mesh = mesh
        # get and set subdomains
        subdomains = MeshFunction("size_t", mesh, subdomains_path)
        self.subdomains = subdomains
        # get and set surfaces
        surfaces = MeshFunction("size_t", mesh, surfaces_path)
        self.surfaces = surfaces

        # create meshes
        self.interior_mesh = MeshView.create(subdomains, 1) # interior mesh
        self.exterior_mesh = MeshView.create(subdomains, 0) # exterior mesh
        self.gamma_mesh = MeshView.create(surfaces, 1)      # interface mesh

        # mark exterior boundary of exterior mesh
        bndry_mesh = MeshView.create(surfaces, 2)       # create mesh for exterior boundary
        bndry_mesh.build_mapping(self.exterior_mesh)    # create mapping between facets
        # access mapping
        facet_mapping = (
            bndry_mesh.topology().mapping()[self.exterior_mesh.id()].cell_map()
        )
        # create exterior boundary mesh function
        self.exterior_boundary = MeshFunction(
            "size_t", self.exterior_mesh, self.exterior_mesh.topology().dim() - 1, 0
        )
        # mark exterior facets with 2
        for i in range(len(facet_mapping)):
            self.exterior_boundary[facet_mapping[i]] = 2

        # define elements
        P1 = FiniteElement("P", self.interior_mesh.ufl_cell(), 1) # potentials
        R0 = FiniteElement("R", self.interior_mesh.ufl_cell(), 0) # Lagrange to enforce /int phi_i = 0
        Q1 = FiniteElement("P", self.gamma_mesh.ufl_cell(), 1)    # membrane ion channels

        # create function spaces
        self.Wi = FunctionSpace(self.interior_mesh, P1)
        self.We = FunctionSpace(self.exterior_mesh, MixedElement([P1, R0]))
        self.Wg = FunctionSpace(self.gamma_mesh, Q1)
        self.W = MixedFunctionSpace(self.Wi, self.We, self.Wg)
        return

    def create_variational_form(self, splitting_scheme=False):
        """ Create a mortar variational formulation for the EMI equations """

        params = self.params
        # get parameters
        dt = params["dt"]                 # global time step (s)
        dt_inv = Constant(1.0 / dt)       # inverted global time step (1/s)
        C_M = params["C_M"]               # capacitance (F/m)
        phi_M_init = params["phi_M_init"] # initial membrane potential (V)
        sigma_i = params["sigma_i"]       # intracellular conductance (S/m^2)
        sigma_e = params["sigma_e"]       # extracellular conductance (S/m^2)
        g_Na_leak = params["g_Na_leak"]   # channel conductance leak (S/m^2)
        g_K_leak = params["g_K_leak"]     # channel conductance leak (S/m^2)
        g_Cl_leak = params["g_Cl_leak"]   # channel conductance leak (S/m^2)
        g_ch_syn = params["g_ch_syn"]     # channel conductance synaptic (S/m^2)
        E_Na = params["E_Na"]             # reversal potential Na (V)
        E_K = params["E_K"]               # reversal potential K (V)
        E_Cl = params["E_Cl"]             # reversal potential Cl (V)

        # set initial membrane potential
        self.phi_M_prev = interpolate(phi_M_init, self.Wg)

        # total channel current
        I_ch = (
            g_Na_leak * (self.phi_M_prev - E_Na)
            + g_ch_syn * (self.phi_M_prev - E_Na)
            + g_K_leak * (self.phi_M_prev - E_K)
            + g_Cl_leak * (self.phi_M_prev - E_Cl)
        )

        # define measures
        dxi = Measure("dx", domain=self.interior_mesh)  # on interior mesh
        dxe = Measure("dx", domain=self.exterior_mesh)  # on exterior mesh
        dxGamma = Measure("dx", domain=self.gamma_mesh) # on interface

        # create functions
        (ui, ue, p_IM) = TrialFunctions(self.W)
        (vi, ve, q_IM) = TestFunctions(self.W)
        self.u_p = Function(self.W)
        ui_p = self.u_p.sub(0)
        ue_p = self.u_p.sub(1)

        # split unknowns
        phi_i = ui
        phi_e, _c = split(ue)
        # split test functions
        vphi_i = vi
        vphi_e, _d = split(ve)

        # initialize Dirichlet boundary conditions
        self.bcs = []

        # initialize all parts of the variational form
        a00 = 0; a01 = 0; a02 = 0; L0 = 0
        a10 = 0; a11 = 0; a12 = 0; L1 = 0
        a20 = 0; a21 = 0; a22 = 0; L2 = 0

        # weak form of equation for phi_i
        a00 += inner(sigma_i * grad(phi_i), grad(vphi_i)) * dxi
        a02 += inner(p_IM, vphi_i) * dxGamma

        # weak form of Lagrange terms (enforcing /int phi_e = 0)
        a11 += _c * vphi_e * dxe
        a11 += _d * phi_e * dxe

        # weak form of equation for phi_e
        a11 += inner(sigma_e * grad(phi_e), grad(vphi_e)) * dxe
        a12 -= inner(p_IM, vphi_e) * dxGamma

        # weak form of equation for phi_M: Lagrange terms
        a20 += inner(phi_i, q_IM) * dxGamma
        a21 -= inner(phi_e, q_IM) * dxGamma
        a22 -= dt / C_M * inner(p_IM, q_IM) * dxGamma
        L2 += inner(self.phi_M_prev, q_IM) * dxGamma
        # add contribution of channel current to equation for phi_M
        if not splitting_scheme:
            L2 -= dt / C_M * inner(I_ch, q_IM) * dxGamma

        # gather var form in matrix structure
        self.a = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
        self.L = L0 + L1 + L2

        # create function for unknown solution
        self.wh = Function(self.W)
        return

    def solve_for_time_step(self):
        """ Solve system for one global time step """

        dt = self.params["dt"]  # global time step (s)

        system = assemble_mixed_system(self.a == self.L, self.wh, self.bcs)
        matrix_blocks = system[0]
        rhs_blocks = system[1]

        AA = PETScNestMatrix(matrix_blocks)
        bb = PETScVector()
        AA.init_vectors(bb, rhs_blocks)
        # Convert VECNEST to standard vector for LU solver (MUMPS doesn't like VECNEST)
        bb = PETScVector(PETSc.Vec().createWithArray(bb.vec().getArray()))
        # Convert MATNEST to AIJ for LU solver
        AA.convert_to_aij()

        # solution vector
        comm = self.exterior_mesh.mpi_comm()
        w = Vector(comm, self.Wi.dim() + self.We.dim() + self.Wg.dim())

        solver = PETScLUSolver()        # create LU solver
        ksp = solver.ksp()              # get ksp solver
        pc = ksp.getPC()                # get pc
        pc.setType("lu")                # set solver to LU
        pc.setFactorSolverType("mumps") # set LU solver to use mumps
        opts = PETSc.Options()          # get options
        opts["mat_mumps_icntl_4"] = 1   # set amount of info output
        opts["mat_mumps_icntl_14"] = 40 # set percentage
        ksp.setFromOptions()            # update ksp with options set above

        solver.solve(AA, w, bb)  # solve system

        # Assign the obtained solution to the function wh defined on the FunctionSpaceProduct
        w0 = Function(self.Wi).vector()
        w0.set_local(w.get_local()[: self.Wi.dim()])
        w0.apply("insert")
        self.wh.sub(0).assign(Function(self.Wi, w0))

        w1 = Function(self.We).vector()
        w1.set_local(w.get_local()[self.Wi.dim() : self.Wi.dim() + self.We.dim()])
        w1.apply("insert")
        self.wh.sub(1).assign(Function(self.We, w1))

        w2 = Function(self.Wg).vector()
        w2.set_local(w.get_local()[self.Wi.dim() + self.We.dim() :])
        w2.apply("insert")
        self.wh.sub(2).assign(Function(self.Wg, w2))

        # update previous ion concentrations
        self.u_p.sub(0).assign(self.wh.sub(0))
        self.u_p.sub(1).assign(self.wh.sub(1))
        self.u_p.sub(2).assign(self.wh.sub(2))
        # update previous membrane potential
        self.phi_M_prev.assign(
            interpolate(self.u_p.sub(0), self.Wg)
            - interpolate(self.u_p.sub(1).sub(0), self.Wg)
        )
        # updates problems time t
        self.t.assign(float(self.t + dt))

        return

    def alpha_n(self, V_M):
        """ Rate coefficient activation Potassium (HH) """

        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 0.01e3 * (10.0 - V) / (exp((10.0 - V) / 10.0) - 1.0)

    def beta_n(self, V_M):
        """ Rate coefficient activation Potassium (HH) """

        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 0.125e3 * exp(-V / 80.0)

    def alpha_m(self, V_M):
        """ Rate coefficient activation Sodium (HH) """

        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 0.1e3 * (25.0 - V) / (exp((25.0 - V) / 10.0) - 1)

    def beta_m(self, V_M):
        """ Rate coefficient activation Sodium (HH) """

        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 4.0e3 * exp(-V / 18.0)

    def alpha_h(self, V_M):
        """ Rate coefficient inactivation Sodium (HH) """
        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 0.07e3 * exp(-V / 20.0)

    def beta_h(self, V_M):
        """ Rate coefficient inactivation Sodium (HH) """
        V_rest = self.params["V_rest"]  # resting potential (V)
        V = 1000 * (V_M - V_rest)       # convert from mV to V

        return 1.0e3 / (exp((30.0 - V) / 10.0) + 1)

    def solve_system_HH(self, n_steps_ode, filename):
        """ Solve KNP-EMI with Hodgkin Huxley dynamics on membrane using a
            splitting scheme """

        # physical parameters
        C_M = self.params["C_M"]             # capacitance (F/m)
        g_Na_bar = self.params["g_Na_bar"]   # Na conductivity Hodgkin Huxley (S/m^2)
        g_K_bar = self.params["g_K_bar"]     # K conductivity Hodgkin Huxley (S/m^2)
        g_K_leak = self.params["g_K_leak"]   # K leak conductivity (S/m^2)
        g_Na_leak = self.params["g_Na_leak"] # Na leak conductivity (S/m^2)
        g_ch_syn = self.params["g_ch_syn"]   # K conductivity Hodgkin Huxley (S/m^2)
        E_Na = self.params["E_Na"]           # Na Nernst potential (V)
        E_K = self.params["E_K"]             # K Nernst potential (V)
        phi_M_init = self.params["phi_M_init"]

        # initial values
        n_init = self.alpha_n(phi_M_init) / (self.alpha_n(phi_M_init) + self.beta_n(phi_M_init))
        m_init = self.alpha_m(phi_M_init) / (self.alpha_m(phi_M_init) + self.beta_m(phi_M_init))
        h_init = self.alpha_h(phi_M_init) / (self.alpha_h(phi_M_init) + self.beta_h(phi_M_init))

        # Hodgkin Huxley parameters
        n = interpolate(Constant(n_init), self.Wg)
        m = interpolate(Constant(m_init), self.Wg)
        h = interpolate(Constant(h_init), self.Wg)

        # total channel currents conductivity
        g_Na = g_Na_leak + g_Na_bar * m ** 3 * h + g_ch_syn
        g_K = g_K_leak + g_K_bar * n ** 4

        # create variational formulation
        self.create_variational_form(splitting_scheme=True)

        # shorthand
        phi_M = self.phi_M_prev
        # total channel current
        I_ch = g_Na * (self.phi_M_prev - E_Na) + g_K * (self.phi_M_prev - E_K)

        # derivatives for Hodgkin Huxley ODEs
        dphidt = -(1 / C_M) * I_ch
        dndt = self.alpha_n(phi_M) * (1 - n) - self.beta_n(phi_M) * n
        dmdt = self.alpha_m(phi_M) * (1 - m) - self.beta_m(phi_M) * m
        dhdt = self.alpha_h(phi_M) * (1 - h) - self.beta_h(phi_M) * h

        # initialize saving results
        save_count = 0
        self.initialize_h5_savefile(filename + "results.h5")
        self.initialize_xdmf_savefile(filename)

        dt_ode = self.dt / n_steps_ode  # ODE time step (s)
        Tstop = self.params["Tstop"]    # global end time (s)
        # solve
        for k in range(int(round(Tstop / float(self.dt)))):
            # Step I: Solve Hodgkin Hodgkin ODEs using backward Euler
            for i in range(n_steps_ode):
                phi_M_new = project(phi_M + dt_ode * dphidt, self.Wg)
                n_new = project(n + dt_ode * dndt, self.Wg)
                m_new = project(m + dt_ode * dmdt, self.Wg)
                h_new = project(h + dt_ode * dhdt, self.Wg)
                assign(phi_M, phi_M_new)
                assign(n, n_new)
                assign(m, m_new)
                assign(h, h_new)
            # Step II: Solve PDEs with phi_M_prev from ODE step
            self.solve_for_time_step()

            # save results
            if (k % 10) == 0:
                save_count += 1
                if save_count == 1:
                    self.save_h5()
                    self.save_xdmf()
                    save_count = 0

            # output to terminal
            mult = 100 / int(round(Tstop / float(self.dt)))
            sys.stdout.write("\r")
            sys.stdout.write("progress: %d%%" % (mult * k))
            sys.stdout.flush()

        # close result files
        self.close_h5()
        self.close_xdmf()

        return

    def initialize_h5_savefile(self, filename):
        """ Initialize h5 file """

        self.h5_idx = 0
        self.h5_file = HDF5File(self.interior_mesh.mpi_comm(), filename, "w")
        self.h5_file.write(self.mesh, "/mesh")
        self.h5_file.write(self.gamma_mesh, "/gamma_mesh")
        self.h5_file.write(self.subdomains, "/subdomains")
        self.h5_file.write(self.surfaces, "/surfaces")

        self.h5_file.write(self.u_p.sub(0), "/interior_solution", self.h5_idx)
        self.h5_file.write(self.u_p.sub(1), "/exterior_solution", self.h5_idx)
        self.h5_file.write(self.phi_M_prev, "/membrane_potential", self.h5_idx)

        return

    def save_h5(self):
        """ Save results to h5 file """

        self.h5_idx += 1
        print(self.h5_idx)
        self.h5_file.write(self.u_p.sub(0), "/interior_solution", self.h5_idx)
        self.h5_file.write(self.u_p.sub(1), "/exterior_solution", self.h5_idx)
        self.h5_file.write(self.phi_M_prev, "/membrane_potential", self.h5_idx)

        return

    def close_h5(self):
        """ Close h5 file """

        self.h5_file.close()

        return

    def initialize_xdmf_savefile(self, file_prefix):
        """ Initialize xdmf files """

        self.interior_xdmf_files = []
        self.exterior_xdmf_files = []

        filename_xdmf = file_prefix + "interior_phi.xdmf"
        xdmf_file = XDMFFile(self.interior_mesh.mpi_comm(), filename_xdmf)
        xdmf_file.parameters["rewrite_function_mesh"] = False
        xdmf_file.parameters["flush_output"] = True
        self.interior_xdmf_files.append(xdmf_file)
        xdmf_file.write(self.u_p.sub(0), self.t.values()[0])

        filename_xdmf = file_prefix + "exterior_phi.xdmf"
        xdmf_file = XDMFFile(self.exterior_mesh.mpi_comm(), filename_xdmf)
        xdmf_file.parameters["rewrite_function_mesh"] = False
        xdmf_file.parameters["flush_output"] = True
        self.exterior_xdmf_files.append(xdmf_file)
        xdmf_file.write(self.u_p.sub(1).split()[0], self.t.values()[0])

        filename_xdmf = file_prefix + "membrane_potential.xdmf"
        self.membrane_xdmf_file = XDMFFile(self.gamma_mesh.mpi_comm(), filename_xdmf)
        self.membrane_xdmf_file.parameters["rewrite_function_mesh"] = False
        self.membrane_xdmf_file.parameters["flush_output"] = True
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])

        return

    def save_xdmf(self):
        """ Save results to xdmf files """

        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].write(self.u_p.sub(0), self.t.values()[0])
            self.exterior_xdmf_files[i].write(
                self.u_p.sub(1).split()[i], self.t.values()[0]
            )
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])

        return

    def close_xdmf(self):
        """ Close xdmf files """

        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].close()
            self.exterior_xdmf_files[i].close()
        self.membrane_xdmf_file.close()

        return
