# Supplementary code Chapter 2 #

Supplementary code for Chapter 2 (A cell-based model for ionic electrodiffusion
in excitable tissue) in EMI: CELL BASED MATHEMATICAL MODEL OF EXCITABLE
CELLS that reproduces Figure 2.1 and 2.2.

### Dependencies ###

Get the environment needed (all dependencies etc.), build and
and run the Docker container *ceciledc/fenics_mixed_dimensional:13-03-20* by:

* Installing docker: https://docs.docker.com/engine/installation/
* Build and start docker container with:

        docker run -t -v ~/:/home/fenics -i ceciledc/fenics_mixed_dimensional:13-03-20

### Usage ###

To run the code and reproduce the results, execute:

        main.py

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
