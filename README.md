# GFVIEM_multilayer
The "k-integrals.py" code contains functions that have overloaded been with numba/numba_scipy, meaning it makes for very fast integration together with the "scipy.integrate.quad" routine.
This makes for faster evaluation of common Sommerfelt integrals needed for calculation of dyadic Green's functions for multilayered reference structures,  as found in e.g T. M. Søndergaard: Green's function integral equation method in Nano-optics.

Also included is a conjugate gradient implementation of the CG-algorithm presented in the same book. 

Thirdly the Build_eps_from_planes.py" contains some obscure code that can eat structures made in Blender such as plane-models, STM-tips or whatever and spits out a dielectric matrix for the structure is also present, but is very user-hostile at the moment.
