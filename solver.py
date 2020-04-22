#!/usr/bin/python3

from dolfin import *
import numpy as np
from petsc4py import PETSc
import sys

from numpy import asarray
from numpy import savetxt

parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

class Solver:
    def __init__(self, ion_list, t, **params):
        """ initialize solver """

        self.ion_list = ion_list    # set ion list
        self.params = params        # set parameters
        self.N_ions = len(ion_list) # set number of ions in ion list
        self.dt = params["dt"]      # set global time step (s)
        self.t = t                  # time constant (s)

        return

    def setup_domain(self, mesh_path, subdomains_path, surfaces_path):
        """ Setup mortar meshes (ECS, ICS, gamma) and element spaces """

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
        bndry_mesh = MeshView.create(surfaces, 2)    # create mesh for exterior boundary
        bndry_mesh.build_mapping(self.exterior_mesh) # create mapping between facets
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
        P1 = FiniteElement(
            "P", self.interior_mesh.ufl_cell(), 1
        )  # ion concentrations and potentials
        R0 = FiniteElement(
            "R", self.interior_mesh.ufl_cell(), 0
        )  # Lagrange to enforce /int phi_i = 0
        Q1 = FiniteElement(
            "P", self.gamma_mesh.ufl_cell(), 1
        )  # total membrane current I_M

        # Intracellular (ICS) ion concentrations for each ion + ICS potential
        interior_element_list = [P1] * (self.N_ions + 1)
        # Extracellular (ECS) ion concentrations for each ion + ECS potential + Lagrange multiplier
        exterior_element_list = [P1] * (self.N_ions + 1) + [R0]

        # create function spaces
        self.Wi = FunctionSpace(self.interior_mesh, MixedElement(interior_element_list))
        self.We = FunctionSpace(self.exterior_mesh, MixedElement(exterior_element_list))
        self.Wg = FunctionSpace(self.gamma_mesh, Q1)
        self.W = MixedFunctionSpace(self.Wi, self.We, self.Wg)

        return

    def create_variational_form(self, splitting_scheme=False, dirichlet_bcs=False):
        """ Create a mortar variation formulation for the KNP-EMI equations """

        params = self.params
        # get parameters
        dt = params["dt"]                   # global time step (s)
        dt_inv = Constant(1.0 / dt)         # invert global time step (1/s)
        F = params["F"]                     # Faraday's constant (C/mol)
        R = params["R"]                     # Gas constant (J/(K*mol))
        temperature = params["temperature"] # temperature (K)
        psi = R * temperature / F           # help variable psi
        C_M = params["C_M"]                 # capacitance (F)
        phi_M_init = params["phi_M_init"]   # initial membrane potential (V)
        m_Na = params["m_Na"]               # threshold pump ICS Na
        m_K = params["m_K"]                 # threshold pump ECS K
        I_max = params["I_max"]             # max pump strength
        g_KCC2 = params["g_KCC2"]           # KCC2 cotransporter
        g_NKCl = params["g_NKCl"]           # NKCl cotransporter

        # set initial membrane potential
        self.phi_M_prev = interpolate(phi_M_init, self.Wg)

        # define measures
        dxe = Measure("dx", domain=self.exterior_mesh)  # on exterior mesh
        dxi = Measure("dx", domain=self.interior_mesh)  # on interior mesh
        dxGamma = Measure("dx", domain=self.gamma_mesh) # on interface

        # measure on exterior boundary
        dsOuter = Measure(
            "ds",
            domain=self.exterior_mesh,
            subdomain_data=self.exterior_boundary,
            subdomain_id=2,
        )

        # define outward normal on exterior boundary (partial Omega)
        self.n_outer = FacetNormal(self.exterior_mesh)

        # create functions
        (ui, ue, p_IM) = TrialFunctions(self.W) # trial functions
        (vi, ve, q_IM) = TestFunctions(self.W)  # test functions
        self.u_p = Function(self.W)             # previous solutions
        ui_p = self.u_p.sub(0)
        ue_p = self.u_p.sub(1)

        # split unknowns
        ui = split(ui)
        ue = split(ue)
        # split test functions
        vi = split(vi)
        ve = split(ve)
        # split previous solution
        ui_prev = split(ui_p)
        ue_prev = split(ue_p)

        # intracellular potential
        phi_i = ui[self.N_ions] # unknown
        vphi_i = vi[self.N_ions] # test function

        # Lagrange multiplier (int phi_e = 0)
        _c = ue[self.N_ions + 1] # unknown
        _d = ve[self.N_ions + 1] # test function

        # extracellular potential
        phi_e = ue[self.N_ions]  # unknown
        vphi_e = ve[self.N_ions] # test function

        # initialize
        alpha_i_sum = 0  # sum of fractions intracellular
        alpha_e_sum = 0  # sum of fractions extracellular
        I_ch = 0         # total channel current
        J_phi_i = 0      # total intracellular flux
        J_phi_e = 0      # total extracellular flux
        self.bcs = []    # Dirichlet boundary conditions

        # Initialize parts of variational formulation
        for idx, ion in enumerate(self.ion_list):
            # get ion attributes
            z = ion["z"]
            Di = ion["Di"]
            De = ion["De"]

            # set initial value of intra and extracellular ion concentrations
            assign(ui_p.sub(idx), interpolate(ion["ki_init"], self.Wi.sub(idx).collapse()))
            assign(ue_p.sub(idx), interpolate(ion["ke_init"], self.We.sub(idx).collapse()))

            # add ion specific contribution to fraction alpha
            ui_prev_g = interpolate(ui_p.sub(idx), self.Wg)
            ue_prev_g = interpolate(ue_p.sub(idx), self.Wg)
            alpha_i_sum += Di * z * z * ui_prev_g
            alpha_e_sum += De * z * z * ue_prev_g

            if dirichlet_bcs:
                # add Dirichlet boundary conditions on exterior boundary
                bc = DirichletBC(self.We.sub(idx), ion["ke_init"], self.exterior_boundary, 2)
                self.bcs.append(bc)

            # calculate and update Nernst potential for current ion
            ion["E"] = project(R * temperature / (F * z) * ln(ue_prev_g / ui_prev_g), self.Wg)

            # get ion channel current
            ion["I_ch"] = ion["g_k"] * (self.phi_M_prev - ion["E"])

        # get ion concentrations at membrane
        Na_i = interpolate(ui_p.sub(0), self.Wg)
        Na_e = interpolate(ue_p.sub(0), self.Wg)
        K_i = interpolate(ui_p.sub(1), self.Wg)
        K_e = interpolate(ue_p.sub(1), self.Wg)
        Cl_i = interpolate(ui_p.sub(2), self.Wg)
        Cl_e = interpolate(ue_p.sub(2), self.Wg)

        # Na/K pump
        self.I_pump = project(I_max / ((1 + m_K / K_e) ** 2 * (1 + m_Na / Na_i) ** 3), self.Wg)

        # KCC2 cotransporter
        KCC2 = ln((K_i * Cl_i) / (K_e * Cl_e))
        self.I_KCC2 = project(g_KCC2 * KCC2, self.Wg)

        # NaKCl2 cotransporter
        NKCl = (1.0 / (1.0 + exp(16.0 - K_e)) * (ln((K_i * Cl_i) / (K_e * Cl_e)) + ln((Na_i * Cl_i) / (Na_e * Cl_e))))
        self.I_NKCl = project(g_NKCl * NKCl, self.Wg)

        # add sodium (Na) contribution
        self.ion_list[0]["I_ch"] += 3 * self.I_pump + self.I_NKCl
        # add potassium (K) contribution
        self.ion_list[1]["I_ch"] += -2 * self.I_pump + self.I_NKCl + self.I_KCC2
        # add chloride (Cl) contribution
        self.ion_list[2]["I_ch"] -= 2 * self.I_NKCl + self.I_KCC2

        # add contribution from each ion to total channel current
        for idx, ion in enumerate(self.ion_list):
            I_ch += ion["I_ch"]

        # Initialize the variational form
        a00 = 0; a01 = 0; a02 = 0; L0 = 0
        a10 = 0; a11 = 0; a12 = 0; L1 = 0
        a20 = 0; a21 = 0; a22 = 0; L2 = 0

        # Setup ion specific part of variational formulation
        for idx, ion in enumerate(self.ion_list):
            # get ion attributes
            z = ion["z"]
            Di = ion["Di"]
            De = ion["De"]
            I_ch_k = ion["I_ch"]

            # Set intracellular ion attributes
            ki = ui[idx]            # unknown
            ki_prev = ui_prev[idx]  # previous solution
            vki = vi[idx]           # test function
            # Set extracellular ion attributes
            ke = ue[idx]            # unknown
            ke_prev = ue_prev[idx]  # previous solution
            vke = ve[idx]           # test function
            # Trace of previous solution on Gamma
            ki_prev_g = interpolate(ui_p.sub(idx), self.Wg)
            ke_prev_g = interpolate(ue_p.sub(idx), self.Wg)
            # Set fraction of ion specific intra--and extracellular I_cap
            alpha_i = Di * z * z * ki_prev_g / alpha_i_sum
            alpha_e = De * z * z * ke_prev_g / alpha_e_sum

            # linearised ion fluxes
            Ji = -Constant(Di) * grad(ki) - Constant(Di * z / psi) * ki_prev * grad(phi_i)
            Je = -Constant(De) * grad(ke) - Constant(De * z / psi) * ke_prev * grad(phi_e)

            # weak form - equation for k_i
            a00 += dt_inv * ki * vki * dxi - inner(Ji, grad(vki)) * dxi
            a02 += 1.0 / (F * z) * alpha_i * p_IM * vki * dxGamma
            L0 += dt_inv * ki_prev * vki * dxi
            L0 -= 1.0 / (F * z) * (I_ch_k - alpha_i * I_ch) * vki * dxGamma

            # weak form - equation for k_e
            a11 += dt_inv * ke * vke * dxe - inner(Je, grad(vke)) * dxe
            a12 -= 1.0 / (F * z) * alpha_e * p_IM * vke * dxGamma
            L1 += dt_inv * ke_prev * vke * dxe
            L1 += 1.0 / (F * z) * (I_ch_k - alpha_e * I_ch) * vke * dxGamma

            # add contribution to total current flux
            J_phi_i += F * z * Ji
            J_phi_e += F * z * Je

        # add contribution from immobile ions to total current flux
        D_A = self.params["D_A"]
        z_A = self.params["z_A"]
        A_i = project(self.params["A_i"], self.Wi.sub(0).collapse())
        A_e = project(self.params["A_e"], self.We.sub(0).collapse())
        J_A_i = -Constant(D_A * z_A / psi) * A_i * grad(phi_i)
        J_A_e = -Constant(D_A * z_A / psi) * A_e * grad(phi_e)
        J_phi_i += F * z_A * J_A_i
        J_phi_e += F * z_A * J_A_e

        # weak form - equation for phi_i
        a00 += inner(J_phi_i, grad(vphi_i)) * dxi
        a02 -= inner(p_IM, vphi_i) * dxGamma

        # weak form - equation for phi_e
        a11 += inner(J_phi_e, grad(vphi_e)) * dxe
        a12 += inner(p_IM, vphi_e) * dxGamma

        # weak form - Lagrange terms (enforcing int phi_e = 0)
        a11 += _c * vphi_e * dxe
        a12 += _d * phi_e * dxe

        # weak form - equation for p_IM
        a20 += inner(phi_i, q_IM) * dxGamma
        a21 -= inner(phi_e, q_IM) * dxGamma
        a22 -= dt / C_M * inner(p_IM, q_IM) * dxGamma
        L2 += inner(self.phi_M_prev, q_IM) * dxGamma
        # add contribution of channel current to equation for phi_M
        if not splitting_scheme:
            L2 += -dt / C_M * inner(I_ch, q_IM) * dxGamma

        # Solution software does not support empty blocks -> hack
        a01 = Constant(0.0)*ke*vki*dxGamma
        a10 = Constant(0.0)*ki*vke*dxGamma

        # gather weak form in matrix structure
        self.a = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
        self.L = L0 + L1 + L2

        # create function for previous (known) solution
        self.wh = Function(self.W)

        return

    def solve_for_time_step(self):
        """ Solve system for one global time step """

        dt = self.params["dt"]                   # global time step (s)
        F = self.params["F"]                     # Faraday's constant (C/mol)
        R = self.params["R"]                     # Gas constant (J/(K mol))
        temperature = self.params["temperature"] # temperature (K)
        m_Na = self.params["m_Na"]               # threshold pump ICS Na
        m_K = self.params["m_K"]                 # threshold pump ECS K
        I_max = self.params["I_max"]             # max pump strength
        g_KCC2 = self.params["g_KCC2"]           # max pump strength
        g_NKCl = self.params["g_NKCl"]           # max pump strength

        # reassemble the block that change in time
        self.matrix_blocks[2] = assemble_mixed(self.alist[2])
        self.matrix_blocks[5] = assemble_mixed(self.alist[5])
        # assemble right hand side
        Llist = extract_blocks(self.L)
        rhs_blocks = [assemble_mixed(L) for L in Llist]# if L is not None]

        AA = PETScNestMatrix(self.matrix_blocks)
        bb = PETScVector()
        AA.init_vectors(bb, rhs_blocks)
        # Convert VECNEST to standard vector for LU solver (MUMPS doesn't like VECNEST)
        bb = PETScVector(PETSc.Vec().createWithArray(bb.vec().getArray()))
        # Convert MATNEST to AIJ for LU solver
        AA.convert_to_aij()

        comm = self.exterior_mesh.mpi_comm()
        w = Vector(comm, self.Wi.dim() + self.We.dim() + self.Wg.dim())

        solver = PETScLUSolver()        # create LU solver
        ksp = solver.ksp()              # get ksp  solver
        pc = ksp.getPC()                # get pc
        pc.setType("lu")                # set solver to LU
        pc.setFactorSolverType("mumps") # set LU solver to use mumps

        opts = PETSc.Options()          # get options
        opts["mat_mumps_icntl_4"] = 1   # set amount of info output
        opts["mat_mumps_icntl_14"] = 40 # set percentage of ???
        ksp.setFromOptions()            # update ksp with options set above

        solver.solve(AA, w, bb)  # solve system

        # assign obtained solution to function wh defined on FunctionSpaceProduct
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
            interpolate(self.u_p.sub(0).sub(self.N_ions), self.Wg)
            - interpolate(self.u_p.sub(1).sub(self.N_ions), self.Wg)
        )

        # updates problems time t
        self.t.assign(float(self.t + dt))

        # update Nernst potential for all ions
        for idx, ion in enumerate(self.ion_list):
            z = ion["z"]
            ki_prev_g = interpolate(self.u_p.sub(0).sub(idx), self.Wg)
            ke_prev_g = interpolate(self.u_p.sub(1).sub(idx), self.Wg)
            updated_E = project(R * temperature / (F * z) * ln(ke_prev_g / ki_prev_g), self.Wg)
            ion["E"].assign(updated_E)

            if idx == 0:
                Na_i = ki_prev_g
                Na_e = ke_prev_g
            if idx == 1:
                K_i = ki_prev_g
                K_e = ke_prev_g
            if idx == 2:
                Cl_i = ki_prev_g
                Cl_e = ke_prev_g

        # update NaK-ATPase pump
        updated_I_pump = project(I_max / ((1 + m_K / K_e) ** 2 * (1 + m_Na / Na_i) ** 3), self.Wg)
        self.I_pump.assign(updated_I_pump)

        # update KCC2 exchnager
        updated_I_KCC2 = project(g_KCC2 * ln((K_i * Cl_i) / (K_e * Cl_e)), self.Wg)
        self.I_KCC2.assign(updated_I_KCC2)

        # NaKCl2 cotransporter
        u_NKCl = (1.0 / (1.0 + exp(16.0 - K_e)) * (ln((K_i * Cl_i) / (K_e * Cl_e)) + ln((Na_i * Cl_i) / (Na_e * Cl_e))))
        updated_I_NKCl = project(g_NKCl * u_NKCl, self.Wg)
        self.I_NKCl.assign(updated_I_NKCl)

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

    def solve_system_HH(self, n_steps_ode, filename, dirichlet_bcs=False):
        """ Solve KNP-EMI with Hodgkin Huxley (HH) dynamics on membrane using a
            two-step splitting scheme """

        # physical parameters
        C_M = self.params["C_M"]                # capacitance (F/m)
        g_Na_bar = self.params["g_Na_bar"]      # Na conductivity HH (S/m^2)
        g_K_bar = self.params["g_K_bar"]        # K conductivity HH (S/m^2)
        V_rest = self.params["V_rest"]          # resting potential (V)
        phi_M_init = self.params["phi_M_init"]  # initial potential (V)

        # initial values
        n_init = self.alpha_n(phi_M_init) / (self.alpha_n(phi_M_init) + self.beta_n(phi_M_init))
        m_init = self.alpha_m(phi_M_init) / (self.alpha_m(phi_M_init) + self.beta_m(phi_M_init))
        h_init = self.alpha_h(phi_M_init) / (self.alpha_h(phi_M_init) + self.beta_h(phi_M_init))

        # Hodgkin Huxley parameters
        n = interpolate(Constant(n_init), self.Wg)
        m = interpolate(Constant(m_init), self.Wg)
        h = interpolate(Constant(h_init), self.Wg)

        # get Na of K from ion list
        Na = self.ion_list[0]
        K = self.ion_list[1]
        Cl = self.ion_list[2]

        # add membrane conductivity of Hodgkin Huxley channels
        Na["g_k"] += g_Na_bar * m ** 3 * h
        K["g_k"] += g_K_bar * n ** 4

        # create variational formulation
        self.create_variational_form(splitting_scheme=True, dirichlet_bcs=dirichlet_bcs)

        # extract the subforms corresponding to each blocks of the formulation
        self.alist = extract_blocks(self.a)
        self.Llist = extract_blocks(self.L)
        # build the variational problem : build needed mappings and sort the BCs
        MixedLinearVariationalProblem(self.alist, self.Llist, self.wh.split(), self.bcs)
        # assemble all the blocks
        self.matrix_blocks = [assemble_mixed(a) for a in self.alist]
        self.rhs_blocks = [assemble_mixed(L) for L in self.Llist]

        # shorthand
        phi_M = self.phi_M_prev
        # derivatives for Hodgkin Huxley ODEs
        dphidt = -(1 / C_M) * (Na["g_k"] * (phi_M - Na["E"])\
                             + K["g_k"] * (phi_M - K["E"]) \
                             + Cl["g_k"] * (phi_M - Cl["E"]) \
                             + self.I_pump)
        dndt = self.alpha_n(phi_M) * (1 - n) - self.beta_n(phi_M) * n
        dmdt = self.alpha_m(phi_M) * (1 - m) - self.beta_m(phi_M) * m
        dhdt = self.alpha_h(phi_M) * (1 - h) - self.beta_h(phi_M) * h

        # initialize saving of results
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
                self.save_h5()
                self.save_xdmf()

            # output to terminal
            mult = 100 / int(round(Tstop / float(self.dt)))
            sys.stdout.write("\r")
            sys.stdout.write("progress: %d%%" % (mult * k))
            sys.stdout.flush()

        # close results files
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

        self.h5_file.write(self.u_p.sub(1), "/exterior_solution", self.h5_idx)
        self.h5_file.write(self.u_p.sub(0), "/interior_solution", self.h5_idx)
        self.h5_file.write(self.phi_M_prev, "/membrane_potential", self.h5_idx)

        return

    def save_h5(self):
        """ Save results to h5 file """

        self.h5_idx += 1
        self.h5_file.write(self.u_p.sub(0), "/interior_solution", self.h5_idx)
        self.h5_file.write(self.u_p.sub(1), "/exterior_solution", self.h5_idx)
        self.h5_file.write(self.phi_M_prev, "/membrane_potential", self.h5_idx)

        return

    def close_h5(self):
        """ Close h5 file """

        self.h5_file.close()

        return

    def initialize_xdmf_savefile(self, file_prefix):
        """ Initialize xdmf file """

        self.interior_xdmf_files = []
        self.exterior_xdmf_files = []
        ion_list_hack = self.ion_list + [{"name": "phi"}]
        for idx, ion in enumerate(ion_list_hack):
            filename_xdmf = file_prefix + "interior_" + ion["name"] + ".xdmf"
            xdmf_file = XDMFFile(self.interior_mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters["rewrite_function_mesh"] = False
            xdmf_file.parameters["flush_output"] = True
            self.interior_xdmf_files.append(xdmf_file)
            xdmf_file.write(self.u_p.sub(0).split()[idx], self.t.values()[0])

            filename_xdmf = file_prefix + "exterior_" + ion["name"] + ".xdmf"
            xdmf_file = XDMFFile(self.exterior_mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters["rewrite_function_mesh"] = False
            xdmf_file.parameters["flush_output"] = True
            self.exterior_xdmf_files.append(xdmf_file)
            xdmf_file.write(self.u_p.sub(1).split()[idx], self.t.values()[0])

        filename_xdmf = file_prefix + "membrane_potential" + ".xdmf"
        self.membrane_xdmf_file = XDMFFile(self.gamma_mesh.mpi_comm(), filename_xdmf)
        self.membrane_xdmf_file.parameters["rewrite_function_mesh"] = False
        self.membrane_xdmf_file.parameters["flush_output"] = True
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])

        return

    def save_xdmf(self):
        """ Save results to xdmf files """

        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].write(self.u_p.sub(0).split()[i], self.t.values()[0])
            self.exterior_xdmf_files[i].write(self.u_p.sub(1).split()[i], self.t.values()[0])
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])

        return

    def close_xdmf(self):
        """ Close xdmf files """

        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].close()
            self.exterior_xdmf_files[i].close()
        self.membrane_xdmf_file.close()

        return
