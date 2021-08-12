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