# Repo for supplementary code #

Supplementary code for Chapter 2 (A cell-based model for ionic electrodiffusion
in excitable tissue) in EMI: CELL BASED MATHEMATICAL MODEL OF EXCITABLE
CELLS (reproduces Figure 2.1 and 2.2).

### Dependencies ###

* Python 3

* The FEniCS Project software: www.fenicsproject.org

  The FEniCS Project is a collection of software for automated
  solution of partial differential equations using finite element
  methods. 

  cbcbeat is compatible with FEniCS version 2017.1.0 and 2017.2.0 (and
  possibly 2016.1.0, 2016.2.0) or development versions in between
  these releases of FEniCS.

To get the environment needed (all dependencies etc.) to run the code, download
and run the following docker container:

`ceciledc/fenics_mixed_dimensional:05-02-20`

To run the code and reproduce the results, execute:

`main.py`

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.


