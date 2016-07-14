rainbow_cfg4.py performance notes
=====================================

Seconds for rainbow 1M 
-----------------------

CAVEAT: using laptop GPU with only 384 cores, desktop GPUs expected x20-30 faster

Disabling step-by-step recording has large improvement
factor for Opticks of about x3 but not much impact on cfg4-.
The result match between G4 and Op remains unchanged.



Seems like reducing the number and size of 
buffers in context is a big win for Opticks.

With step by step and sequence recording::

   Op   4.6    5.8      # Opticks timings rather fickle to slight code changes, maybe stack 
   G4   56.8  55.9

Just final photon recording:: 

   Op    1.8 
   G4   47.9



CAVEAT: above Op is Op/INTEROP
--------------------------------

Actually this behavior is for Opticks INTEROP mode using OpenGL
buffers, in compute mode with OptiX buffers there is almost no
difference between enabling step-by-step recording and not. 
It seems like OpenGL constrains performance once
total buffer size gets too big.



Matching curand buffer to requirement
---------------------------------------

* tried using 1M cuRAND buffer matching the requirement rather than using default 3M all the time,
  saw no change in propagation time 

::

    # change ggeoview-rng-max value down to 1M

    ggeoview-rng-prep  # create the states cache 
 
    #  opticks-/OpticksCfg.hh accordingly 


Compute Mode, ie no OpenGL
-----------------------------

Revived "--compute" mode of ggv binary which uses OptiX owned buffers
as opposed to the usual interop approach of using OpenGL buffers.
Both with and without step recording is giving similar times in 
compute mode. This is very different from interop mode where 
cutting down on buffers gives big wins.

::

    Op   0.75  0.65 
    G4   57.   56.

A related cmp mode controlled by "--cmp" option uses different computeTest binary, 
is not operational and little motivation now that "--compute" mode works.
Could create package without OpenGL dependencies if there is a need.

::

   ggv-;ggv-rainbow --compute 
   ggv-;ggv-rainbow --compute --nostep 
   ggv-;ggv-rainbow --compute --nostep --dbg


* look at how time scales with photon count  


Split the prelaunch from launch timings
-----------------------------------------

Kernel validation, compilation and prelaunch does not 
need to be done for each event so can exclude it from 
timings. 

Doing this get::

    Op (interop mode)         1.509 
    Op (--compute)            0.290
    Op (--compute --nostep)   0.294     # skipping step recording not advantageous   
    Op (--compute)            0.1416    # hmm some performance instability

In ">console" login mode "ggv-rainbow" gives error that no GPU available

Immediately after login getting::

    Op (--compute)            0.148


Testing in Console Mode
-------------------------

::

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_170136/t_delta.ini:propagate=0.14798854396212846

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171121/t_delta.ini:propagate=0.44531063502654433  # try >console mode 
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171142/t_delta.ini:propagate=0.45501201006118208
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171156/t_delta.ini:propagate=0.33855076995678246 
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171213/t_delta.ini:propagate=0.46851423906628042
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171226/t_delta.ini:propagate=0.33861030195839703

    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171527/t_delta.ini:propagate=1.5933509200112894   # GUI interop mode
    /usr/local/env/opticks/rainbow/mdtorch/5/20160102_171548/t_delta.ini:propagate=0.27229616406839341  # GUI --compute mode

Immediately after switching back to automatic graphics switching, then shortly after that::

    0.142      
    0.293



To do the nostep check
------------------------

After standard comparison::

   ggv-;ggv-rainbow 
   ggv-;ggv-rainbow --cfg4 

* recompile optixrap- without RECORD define 
* run with --nostep option::

   ggv-;ggv-rainbow --nostep 
   ggv-;ggv-rainbow --cfg4 --nostep 




