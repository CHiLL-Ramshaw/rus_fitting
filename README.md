# rus_fitting
- Code to fit the full elastic tensor to a resonant ultrasound spectroscopy resonance spectrum.
- The code consists of forward solvers and fitting classes. We have implemented three forward solvers (RPR, SMI, Comsol) and three fitting classes (ray, lmfit, scipy_leastsq).
- A typical workflow for a fit looks as follows:
    - initialize a forward solver (or "rus_object")
    - pass the rus_object to the fitting class

More details on all forward solvers and fitting classes are given below:

# Forward Solvers
## RPR
The RPR forward solver can only compute resonance spectra for geometries in the shape of rectangular parallelepipeds. A typical workflow looks as follows:
- create the potential and kinetic energy matrices with "rpr_matrices.py"
    - you can save those as .npz files, so that you only have to do this step once; if you want to re-run a fit later, those can simply be loaded and this step can be skipped
- create an RUS object by loading the energy matrices with "rus_xyz.py"
- pass the RUS object to a fitting class and run the fit

## SMI
The SMI forward solver can compute resonance spectra for irregularly shaped samples. A 3D surface mesh of the geometry is required for this method. A typical workflow looks as follows:
- create the potential and kinetic energy matrices with "smi_matrices.py"
    - you can save those as .npz files, so that you only have to do this step once; if you want to re-run a fit later, those can simply be loaded and this step can be skipped
- create an RUS object by loading the energy matrices with "rus_xyz.py"
- pass the RUS object to a fitting class and run the fit

## Comsol
The Comsol forward solver can compute resonance spectra for irregularly shaped samples. A comsol .mph file with the correct physics module needs to be loaded. A typical workflow looks as follows:
- create an RUS object with "rus_comsol.py"
- pass the RUS object to a fitting class and run the fit