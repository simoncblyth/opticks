U4RecorderTest_U4Stack_stag_enum_random_alignment
===================================================

Overview
---------

Opticks
   qsim.h stagr.h stag.h 
Geant4 
   U4Recorder U4Stack::Classify SBacktrace.h  

Machinery for enumerating and collecting random consumption records 
in both contexts is essentially complete following the ideas from prior. 

* :doc:`ideas_on_random_alignment_in_new_workflow`

Note that if there is a need to annotate the "simstreams" to help with
alignment : can do that simply by adding burns with suitable enum names. 

Need to apply the machinery to input_photons with a variety of
propagation histories to observe the consumption patterns
in order to decide how best to align. 



WIP : apply consumption enum collection machinery with storch_test.sh input photons
-----------------------------------------------------------------------------------------

::

    cx
    ./cxs_raindrop.sh       # remote 
    ./cxs_raindrop.sh grab  # local 

    u4t
    ./U4RecorderTest.sh     # local 

    ## NB without the below envvar U4RecorderTest does not fill the stack tags, leaving them all zero
    ##  export U4Random_flat_debug=1  

    u4t
    ./U4RecorderTest_ab.sh     # local 
     




Unaligned initial small geometry
----------------------------------

::

    In [17]: seqhis_(a.seq[:6,0])
    Out[17]: 
    ['TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BR SA']

    In [18]: seqhis_(b.seq[:6,0])
    Out[18]: 
    ['TO BT BT SA',
     'TO BT BT SA',
     'TO BT BT SA',
     'TO BR SA',
     'TO BT BT SA',
     'TO BT BT SA']

    ## when the flat are there they match 

    In [15]: a.flat[:6,:14]
    Out[15]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.   , 0.   ],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.509, 0.089, 0.007, 0.954, 0.   , 0.   ],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.082, 0.807, 0.695, 0.618, 0.   , 0.   ],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.491, 0.328, 0.911, 0.191, 0.   , 0.   ],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.079, 0.148, 0.599, 0.426, 0.   , 0.   ],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]], dtype=float32)


    In [16]: b.flat[:6,:14]
    Out[16]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.   , 0.   ],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.509, 0.089, 0.007, 0.954, 0.   , 0.   ],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.082, 0.807, 0.695, 0.618, 0.   , 0.   ],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.491, 0.328, 0.   , 0.   , 0.   , 0.   ],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.079, 0.148, 0.599, 0.426, 0.   , 0.   ],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.361, 0.62 , 0.45 , 0.306, 0.   , 0.   ]], dtype=float32)


    In [13]: at[:6, :14]
    Out[13]: 
    array([[ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0],
           [ 1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0]], dtype=uint8)


    # A: step preamble deciding which process wins is 1,2 

    In [9]: print(tag.label(at[0,:14]))
     0 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     1 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     2 :  9 :      at_bo : boundary burn 
     3 : 10 :      at_rf : u_reflect > TransCoeff 
     4 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     5 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     6 :  9 :      at_bo : boundary burn 
     7 : 10 :      at_rf : u_reflect > TransCoeff 
     8 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     9 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
    10 : 11 :      sf_sd : qsim::propagate_at_surface ab/sd 
    11 : 12 :      sf_bu : qsim::propagate_at_surface burn 
    12 :  0 :      undef : undef 
    13 :  0 :      undef : undef 

    In [10]: print(tag.label(at[5,:14]))
     0 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     1 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     2 :  9 :      at_bo : boundary burn 
     3 : 10 :      at_rf : u_reflect > TransCoeff 
     4 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     5 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     6 : 11 :      sf_sd : qsim::propagate_at_surface ab/sd 
     7 : 12 :      sf_bu : qsim::propagate_at_surface burn 
     8 :  0 :      undef : undef 
     9 :  0 :      undef : undef 
    10 :  0 :      undef : undef 
    11 :  0 :      undef : undef 
    12 :  0 :      undef : undef 
    13 :  0 :      undef : undef 

    In [14]: bt[:6, :14]
    Out[14]: 
    array([[2, 6, 4, 3, 8, 7, 2, 6, 8, 7, 2, 6, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 8, 7, 2, 6, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 8, 7, 2, 6, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 2, 6, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 8, 7, 2, 6, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 8, 7, 2, 6, 0, 0]], dtype=uint8)

    # step preamble deciding on winner process is 2,6,4,3 
    # BUT that does not fully re-run for each step getting only 2,6 for subsequent


    In [19]: print(stack.label(bt[0,:14]))
     0 :  2 : ScintDiscreteReset :  
     1 :  6 : BoundaryDiscreteReset :  
     2 :  4 : RayleighDiscreteReset :  
     3 :  3 : AbsorptionDiscreteReset :  

     4 :  8 : BoundaryBurn :  
     5 :  7 : BoundaryDiDi :  

     6 :  2 : ScintDiscreteReset :  
     7 :  6 : BoundaryDiscreteReset :  

     8 :  8 : BoundaryBurn :  
     9 :  7 : BoundaryDiDi :  

    10 :  2 : ScintDiscreteReset :  
    11 :  6 : BoundaryDiscreteReset :  
    12 :  0 : Unclassified :  
    13 :  0 : Unclassified :  



DONE : observe how consumption changes when use U4Process::ClearNumberOfInteractionLengthLeft 
--------------------------------------------------------------------------------------------------

* U4Process::ClearNumberOfInteractionLengthLeft called from tail of U4Recorder::UserSteppingAction_Optical

::

    182 void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
    183 {
    ...
    258     if( tstat == fAlive )
    259     {
    260         U4Process::ClearNumberOfInteractionLengthLeft(*track, *step);
    261     }
    262 


Q: This will inevitably change the simulation because are using 
   different randoms, but does it change the correctness of the simulation ?

A: Assuming just technical change, because the chances of SC/AB etc..
   are surely independent of what happened before ? 


* with this the step point preamble now 2,6,4,3 with all 4 process reset for every step point
* the advantage of this is its simplicity and similarity of each step point 

* the preamble consumption can be regarded as the arrows between flag points, 
  that act to decide what the next history flag will be::

  TO->BT->BT->SA 

* where does SA fit into this ? B:G4 is getting NoRINDEX truncated ?
  but A actually finds perfectAbsorbSurface boundary

* TODO:Geant4 surface equivalent on the Rock///Air boundary  


::

    In [6]: bt[:5,:20]
    Out[6]: 
    array([[2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0]], dtype=uint8)


    In [2]: print(stack.label(bt[0,:20]))
     0 :  2 : ScintDiscreteReset :  
     1 :  6 : BoundaryDiscreteReset :  
     2 :  4 : RayleighDiscreteReset :  
     3 :  3 : AbsorptionDiscreteReset :  
     4 :  8 : BoundaryBurn :  
     5 :  7 : BoundaryDiDi :  

     6 :  2 : ScintDiscreteReset :  
     7 :  6 : BoundaryDiscreteReset :  
     8 :  4 : RayleighDiscreteReset :  
     9 :  3 : AbsorptionDiscreteReset :  
    10 :  8 : BoundaryBurn :  
    11 :  7 : BoundaryDiDi :  

    12 :  2 : ScintDiscreteReset :  
    13 :  6 : BoundaryDiscreteReset :  
    14 :  4 : RayleighDiscreteReset :  
    15 :  3 : AbsorptionDiscreteReset :  

    16 :  0 : Unclassified :  
    17 :  0 : Unclassified :  
    18 :  0 : Unclassified :  
    19 :  0 : Unclassified :  

    In [4]: at[:5,:20]
    Out[4]: 
    array([[ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)


    TO->BT->BT->SA 

    In [5]: print(tag.label(at[0,:20]))
     0 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     1 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     2 :  9 :      at_bo : boundary burn 
     3 : 10 :      at_rf : u_reflect > TransCoeff 

     4 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     5 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 
     6 :  9 :      at_bo : boundary burn 
     7 : 10 :      at_rf : u_reflect > TransCoeff 

     8 :  1 :      to_sc : qsim::propagate_to_boundary u_scattering 
     9 :  2 :      to_ab : qsim::propagate_to_boundary u_absorption 

    10 : 11 :      sf_sd : qsim::propagate_at_surface ab/sd 
    11 : 12 :      sf_bu : qsim::propagate_at_surface burn 

    12 :  0 :      undef : undef 
    13 :  0 :      undef : undef 
    14 :  0 :      undef : undef 
    15 :  0 :      undef : undef 
    16 :  0 :      undef : undef 
    17 :  0 :      undef : undef 
    18 :  0 :      undef : undef 
    19 :  0 :      undef : undef 


* adding two burns at step front to A would bring them into line 
* at_surface difference at the end due to the NoRINDEX Rock trick probably ?

  * TODO: ADD A GEANT4 SURFACE TO THE TEST GEOMETRY TO GET THE TAIL TO ALIGN









Try with::

    182 void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
    183 {
    ...
    258     //if( tstat == fAlive )
    259     {
    260         U4Process::ClearNumberOfInteractionLengthLeft(*track, *step);
    261     }
    262 
    263 
    264 }

Seems no difference, presumably all fAlive ?::

    In [1]: bt[:5,:20]
    Out[1]: 
    array([[2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 0, 0, 0, 0]], dtype=uint8)








TODO : observe with bigger geometry so can see AB and SC in the history 
--------------------------------------------------------------------------



DONE : checked storch_test.sh MOCK_CURAND input photons match on laptop and workstation
------------------------------------------------------------------------------------------

Confirmed perfect match with input photons generated on Linux workstation and Apple laptop::

    cd ~/opticks/sysrap/tests
    ./storch_test.sh       # remote  
    ./storch_test.sh       # local  
    ./storch_test.sh grab  # local  
    ./storch_test.sh cf  # local using sysrap/tests/storch_test_cf.py    


