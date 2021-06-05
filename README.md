# GFVIEM_multilayer
This code contains functions that have overloaded with numba, meaning it talks together with the "scipy.integrate.quad" routine.
This makes for faster evaluation of common Sommerfelt integrals needed for calculation of dyadic Green's functions for multilayered reference structures,  as found in e.g T. M. Søndergaard: Green's function integral equation method in Nano-optics.

Also included is a conjugate gradient implementation of the CG-algorithm presented in the same book. 

Thirdly some obscure code that can eat structures made in Blender such as plane-models and spits out a dielectric matrix for the structure is also present, but hardly useable by anybody at the moment.
