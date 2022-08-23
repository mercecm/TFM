This is the GitHub repository of the computational material for my Master's Thesis "Solving the transport equations of nanoparticles in solution using FEniCS."

In folder "Variable Mesh" can be found the code that contains the function that creates non constant meshes and some examples in 1D and 2D.

In folder "Stationary Solution" can be found the code of section 3.1. In the columns of the csv files are stored x coordinate, u computed, u exact, flux computed, flux exact. Figures shown in the written work can also be found in the folders mu0.01 and mu0.1, with the python codes that generate them from the csv output files.

In folder "Time dependant solution" can be found the code of section 3.2, "tDependant.py". "exact.py" and "plots.py" generate the plots shown in the written work from the csv output files.

In folder "2D" one can find the codes that generate the stationary solution and the time dependant solution of section 3.3. For each case, PBC means that Periodic Boundary Conditions are set and NBC means that von Neumann Boundary Conditions are set.

In folder "Coupling" the code that solves the coupled system of section 4 can be found. "DA-NS_1.py" solves the cases with constant magnetic field gradient and "DA-NS_2.py" solves the case with non-constant magnetic field gradient.
