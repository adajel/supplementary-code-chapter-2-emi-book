#!/usr/bin/env python3

"""
This script generates a 3D mesh representing 2 axons.
"""

from dolfin import *
import sys


class Boundary(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return on_boundary


def add_axon(mesh, subdomains, surfaces, a, b):
    # define interior domain
    in_interior = """ (x[0] >= %g && x[0] <= %g &&
                       x[1] >= %g && x[1] <= %g &&
                       x[2] >= %g && x[2] <= %g) """ % (
        a[0],
        b[0],
        a[1],
        b[1],
        a[2],
        b[2],
    )

    interior = CompiledSubDomain(in_interior)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        subdomains[cell] += int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        side_1 = near(x[0], a[0]) and a[1] <= x[1] <= b[1] and a[2] <= x[2] <= b[2]
        side_2 = near(x[0], b[0]) and a[1] <= x[1] <= b[1] and a[2] <= x[2] <= b[2]
        side_3 = near(x[1], a[1]) and a[0] <= x[0] <= b[0] and a[2] <= x[2] <= b[2]
        side_4 = near(x[1], b[1]) and a[0] <= x[0] <= b[0] and a[2] <= x[2] <= b[2]
        side_5 = near(x[2], a[2]) and a[0] <= x[0] <= b[0] and a[1] <= x[1] <= b[1]
        side_6 = near(x[2], b[2]) and a[0] <= x[0] <= b[0] and a[1] <= x[1] <= b[1]
        surfaces[facet] += side_1 or side_2 or side_3 or side_4 or side_5 or side_6

    return


# if no input argument, set resolution factor to default
if len(sys.argv) == 1:
    resolution_factor = 0
else:
    resolution_factor = int(sys.argv[1])

l = 1
nx = l * 160 * 2 ** resolution_factor
ny = 11 * 2 ** resolution_factor
nz = 8 * 2 ** resolution_factor

# box mesh
mesh = BoxMesh(Point(0, 0.0, 0.0), Point(l * 200, 1.1, 0.8), nx, ny, nz)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
# mark exterior boundary
Boundary().mark(surfaces, 2)

a = Point(5, 0.3, 0.3)
b = Point(l * 200 - 5, 0.5, 0.5)
add_axon(mesh, subdomains, surfaces, a, b)

a = Point(5, 0.6, 0.3)
b = Point(l * 200 - 5, 0.8, 0.5)
add_axon(mesh, subdomains, surfaces, a, b)

# convert mesh to unit meter (m)
mesh.coordinates()[:, :] *= 1e-6

# path to directory where mesh files are saved
dir_path = "meshes/" + "two_neurons_3d/"

meshfile = File(dir_path + "mesh_" + str(resolution_factor) + ".xml")
meshfile << mesh

subdomains_file = File(dir_path + "subdomains_" + str(resolution_factor) + ".xml")
subdomains_file << subdomains

surfaces_file = File(dir_path + "surfaces_" + str(resolution_factor) + ".xml")
surfaces_file << surfaces

meshplot = File(dir_path + "subdomains_" + str(resolution_factor) + ".pvd")
meshplot << subdomains

surfacesplot = File(dir_path + "surfaces_" + str(resolution_factor) + ".pvd")
surfacesplot << surfaces
