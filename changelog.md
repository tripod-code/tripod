### Things I have changed in the python vesrsion to make it consistent with PLUTO

- changed the shrinkage tertm to forcce the solution to f = fcrit at each timestep -> makes the results te same as in pluto 

- changed the power law to be consistent wit hthe tripod code sent to me -> different than in the paper -> switch the two turbulent regimes
-> changed how the transition berween driftfag and turb frag is calcualted (now same as in tripod)

- changed the relative velocity terms for the browninan motion (only considders the larger size), radial drift (uses vmax *.. instead of vdrift of the code) and vertical settleing (st/1+st**2 instead of cutoff) to be consistent with PLuto 

-changed the way the drift fuge factor is implemented -> modify the stokes number in the calculation of the drift and diffusion (to be consistent with tripod)

-for now coagualtion is added as implicit term and added to the matrix inversion 

-added the reconstructed power law qeff used in the calcualtion to compute the representative sizes 


#### things beyond the PLUTO version that have been changhed 


#### Ideas what should be adapted 
- timestep for smax does not considder the advection 

- timestep when shrinking is not completely working 
    - need to calculate the balance between the hydro source term of smax and the shrinkage -> could calculate the proper  timestep that way 
    - with his timestep calculation we could add the smax shrinakge as source again
    
