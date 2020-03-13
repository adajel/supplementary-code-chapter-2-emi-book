# Repo for supplementary code #

Supplementary code for Chapter 2 (A cell-based model for ionic electrodiffusion
in excitable tissue) in EMI: CELL BASED MATHEMATICAL MODEL OF EXCITABLE
CELLS (reproduces Figure 2.1 and 2.2).

### Dependencies and ###

* Python 3

* The FEniCS Project software: www.fenicsproject.org
> The FEniCS Project is a collection of software for automated
  solution of partial differential equations using finite element
  methods. This software is compatible with FEniCS version xxxx.

Get the environment needed (all dependencies etc.), build and
and run the Docker container *ceciledc/fenics_mixed_dimensional:05-02-20* by:

* Installing docker: https://docs.docker.com/engine/installation/
* Build and start docker container with 
`docker run -t -v ~/:/home/fenics -i ceciledc/fenics_mixed_dimensional:05-02-20`

To run the code and reproduce the results, execute:

`main.py`

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.


