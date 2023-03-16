Here we demonstrate RUS fits with different methods:

Fit data:
- All fits are using the same resonance frequencies given in "ResonanceList.dat"
- this input file should contain frequencies in MHz in the first column and a weight in the second column
- weight 0 will exclude that resonance from the fit
- if your fit input file only contains frequencies, but no weights you can use "format_resonances_for_fit.py"

We give examples of fitting with:
 - the RPR method and the rus_fitting_lmfit code
 - the SMI method and the rus_fitting_ray code
 - the comsol method and the rus_fitting_scipy_leastsq code