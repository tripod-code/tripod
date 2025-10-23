### Things I have changed in the python version to make it consistent with PLUTO

- changed the shrinkage tertm to force the solution to f = fcrit at each timestep -> makes the results te same as in pluto 

- changed the power law to be consistent with the tripod code sent to me -> different than in the paper -> switch the two turbulent regimes

- changed how the transition berween driftfag and turb frag is calcualted (now same as in tripod)

- changed the way the drift fuge factor is implemented -> modify the stokes number in the calculation of the drift and diffusion (to be consistent with tripod)

- for now coagualtion is added as implicit term and added to the matrix inversion 

- added the reconstructed power law qeff used in the calcualtion to compute the representative sizes 

#### things beyond the PLUTO version that have been changhed 

- added tracers and components that can sublimate and condensate between gas and dust

#### Ideas what should be adapted/Future directions

- Solve all the components in the same matrix to make chemistry possible

- find a way to implictly calcualte the growth of amax
    
