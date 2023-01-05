# rus_comsol
Fit RUS data using COMSOL

# To import in COMSOL ct-scans without weird boundaries that cause problems
# for meshes:
# 1) Import the stl file
# 2) Global Definitions> Mesh Parts> Mesh Part 1>Import
#    In Boundary conditioning, select Minimal
# This should remove the weird boundaries.
# The other solution is to import the stl file in Blender and to use a smooth
# modifier.


# for stokes matrices
We use quadpy to calculate integrals of the triangles.
For correct installation of quadpy:
1) you might have to: conda update --all (in anaconda prompt)
the rest is in command prompt
2) pip uninstall quadpy
3) pip uninstall orthopy
4) pip install quadpy --force-reinstall


Then in the source code of quadpy, in folder "tn", in file "_helpers.py":
replace get_vol function with:

get_vol(simplex):
    x1 = simplex[1] - simplex[0]
    x2 = simplex[2] - simplex[0]
    vol =  0.5*np.absolute(np.cross(x1, x2))
    return vol

# Note: newer version of quadpy don't allow to access source code anymore; v0.16.10 should work