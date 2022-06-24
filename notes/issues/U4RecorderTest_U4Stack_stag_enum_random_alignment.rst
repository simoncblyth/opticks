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
     

TODO : scripted tabulation of the A:tags and B:stacks with U4RecorderTest_ab.py to use while effecting the alignment
-----------------------------------------------------------------------------------------------------------------------


TODO : adjust how StepTooSmall is handled to avoid messing up the consumption regularity 
---------------------------------------------------------------------------------------------

* HMM in CFG4 I recall doing some jump backs to stay aligned. Was that for StepTooSmall ?


DONE : folding A:tags and B:stacks arrays for clarity and easier querying using stag.StepSplit 
---------------------------------------------------------------------------------------------------
::

    In [3]: seqhis_(a.seq[:5,0])
    Out[3]: ['TO BT BT SA', 'TO BT BT SA', 'TO BT BT SA', 'TO BT BT SA', 'TO BT BT SA']

    In [4]: seqhis_(b.seq[:5,0])
    Out[4]: ['TO BT BT SA', 'TO BT BT SA', 'TO BT BT SA', 'TO BR SA', 'TO BT BT SA']


Consumption pattern expected to always have same start to each steppoint from the stack Reset deciding
on what process will win the step.  So rearranging array into those steps makes it easier to follow and query::

    In [8]: at[:5,:20]   # A:tags
    Out[8]: 
    array([[ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)

    In [9]: bt[:5,:20]   # B:stacks
    Out[9]: 
    array([[2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 2, 6, 8, 9, 0, 0, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0]], dtype=uint8)

::

    In [10]: at[0]
    Out[10]: array([ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=uint8)

::

    In [18]: starts = np.where( at[0] == 1 )[0] ; starts
    Out[18]: array([0, 4, 8])

    ends = np.where( at[0] == 0 )   
    end = ends[0][0] 

    In [21]: at[0,0:4]
    Out[21]: array([ 1,  2,  9, 10], dtype=uint8)

    In [22]: at[0,4:8]
    Out[22]: array([ 1,  2,  9, 10], dtype=uint8)

    In [56]: at[0,8:end]
    Out[56]: array([ 1,  2, 11, 12], dtype=uint8)

    ats = np.zeros( (5, 10), dtype=np.uint8 ) 
    ats[0,0:4] = at[0,0:4]  
    ats[1,0:4] = at[0,4:8]  
    ats[2,0:4] = at[0,8:end]   


stag.py::

     41     @classmethod
     42     def StepSplit(cls, tg, step_slot=10):
     43         """
     44         :param tg: unpacked tag array of shape (n, SLOTS)
     45         :param step_slot: max random throws per step  
     46         :param tgs: step split tag array of shape (n, max_step, step_slot) 
     47 
     48         In [4]: at[0]
     49         Out[4]: array([ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12,  0,  0,  0,  0], dtype=uint8)
     50 
     51         In [8]: ats[0]
     52         Out[8]: 
     53         array([[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
     54                [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
     55                [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0],
     56                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=uint8)
     57 
     58         """
     59 
     60         max_starts = 0
     61         for i in range(len(tg)):
     62             starts = np.where( tg[i] == tg[0,0] )[0]
     63             if len(starts) > max_starts: max_starts = len(starts)
     64         pass
     65         
     66         tgs = np.zeros((len(tg), max_starts, step_slot), dtype=np.uint8)
     67         for i in range(len(tg)): 
     68             starts = np.where( tg[i] == tg[0,0] )[0]
     69             ends = np.where( tg[i] == 0 )[0] 
     70             end = ends[0] if len(ends) > 0 else len(tg[i])   ## handle when dont get zero due to truncation
     71             for j in range(len(starts)):
     72                 st = starts[j]
     73                 en = starts[j+1] if j+1 < len(starts) else end
     74                 tgs[i, j,0:en-st] = tg[i,st:en] 
     75             pass
     76         pass
     77         return tgs



Difficult to interpret whats happening when have truncation::

    In [2]: ats[53]
    Out[2]: 
    array([[ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
           [ 1,  2,  9, 10,  0,  0,  0,  0,  0,  0],
           [ 1,  2, 11, 12,  0,  0,  0,  0,  0,  0]], dtype=uint8)

    In [3]: bts[53]
    Out[3]: 
    array([[2, 6, 4, 3, 8, 7, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 0, 0, 0, 0],
           [2, 6, 4, 3, 0, 0, 0, 0, 0, 0],
           [2, 6, 8, 7, 0, 0, 0, 0, 0, 0],
           [2, 6, 4, 3, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    In [4]: seqhis_(a.seq[53,0])
    Out[4]: 'TO BT BR BR BR BT SA'

    In [5]: seqhis_(b.seq[53,0])
    Out[5]: 'TO BT BR BT SA'

    In [6]: at[53]
    Out[6]: array([ 1,  2,  9, 10,  1,  2,  9, 10,  1,  2,  9, 10,  1,  2,  9, 10,  1,  2,  9, 10,  1,  2, 11, 12], dtype=uint8)

    In [7]: bt[53]
    Out[7]: array([2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 2, 6, 8, 7, 2, 6, 4, 3], dtype=uint8)


    In [1]: print(stack.label(bt[53]))
     0 :  2 : ScintDiscreteReset :   
     1 :  6 : BoundaryDiscreteReset :   
     2 :  4 : RayleighDiscreteReset :   
     3 :  3 : AbsorptionDiscreteReset :   
     4 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
     5 :  7 : BoundaryDiDiTransCoeff :   

     6 :  2 : ScintDiscreteReset :   
     7 :  6 : BoundaryDiscreteReset :   
     8 :  4 : RayleighDiscreteReset :   
     9 :  3 : AbsorptionDiscreteReset :   
    10 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
    11 :  7 : BoundaryDiDiTransCoeff :   

    12 :  2 : ScintDiscreteReset :   
    13 :  6 : BoundaryDiscreteReset :   
    14 :  4 : RayleighDiscreteReset :   
    15 :  3 : AbsorptionDiscreteReset :   

    16 :  2 : ScintDiscreteReset :   
    17 :  6 : BoundaryDiscreteReset :   
    18 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
    19 :  7 : BoundaryDiDiTransCoeff :   
    ##  HMM: ONLY 2 RESET, NOT NORMAL GANG OF 4 ?

    20 :  2 : ScintDiscreteReset :   
    21 :  6 : BoundaryDiscreteReset :   
    22 :  4 : RayleighDiscreteReset :   
    23 :  3 : AbsorptionDiscreteReset :   

How often ? 8/100::

    In [9]: np.where(bts[:,:,2] == 8 )
    Out[9]: (array([ 3, 15, 21, 25, 36, 53, 54, 64]), array([2, 2, 2, 2, 3, 3, 3, 3]))

    In [10]: bts[3]
    Out[10]: 
    array([[2, 6, 4, 3, 8, 7, 0, 0, 0, 0],
           [2, 6, 4, 3, 0, 0, 0, 0, 0, 0],
           [2, 6, 8, 9, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    In [11]: bts[15]
    Out[11]: 
    array([[2, 6, 4, 3, 8, 7, 0, 0, 0, 0],
           [2, 6, 4, 3, 0, 0, 0, 0, 0, 0],
           [2, 6, 8, 9, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)


Whats special about those 8 ? All have StepTooSmall skip outs::

    2022-06-24 12:20:06.817 INFO  [30005984] [U4RecorderTest::GeneratePrimaries@119] ]
    2022-06-24 12:20:06.817 INFO  [30005984] [U4Recorder::BeginOfEventAction@52] 
    2022-06-24 12:20:07.123 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.124 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 64
    2022-06-24 12:20:07.214 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.214 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 54
    2022-06-24 12:20:07.227 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.227 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 53
    2022-06-24 12:20:07.379 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.379 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 36
    2022-06-24 12:20:07.476 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.476 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 25
    2022-06-24 12:20:07.509 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.509 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 21
    2022-06-24 12:20:07.561 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.561 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 15
    2022-06-24 12:20:07.666 ERROR [30005984] [U4StepPoint::Flag@123]  fGeomBoundary  U4OpBoundaryProcessStatus::Name StepTooSmall flag NAN_ABORT
    2022-06-24 12:20:07.666 ERROR [30005984] [U4Recorder::UserSteppingAction_Optical@209]  skipping StepTooSmall label.id 3
    2022-06-24 12:20:07.693 INFO  [30005984] [U4Recorder::EndOfEventAction@53] 
    2022-06-24 12:20:07.693 INFO  [30005984] [U4Recorder::EndOfRunAction@51] 


Increase stag.h/stag.py:NSEQ to 4 increases SLOTS to 48, avoiding truncation::

    In [3]: print(stack.label(bt[53,:27]))
     0 :  2 : ScintDiscreteReset :   
     1 :  6 : BoundaryDiscreteReset :   
     2 :  4 : RayleighDiscreteReset :   
     3 :  3 : AbsorptionDiscreteReset :   
     4 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
     5 :  7 : BoundaryDiDiTransCoeff :   

     6 :  2 : ScintDiscreteReset :   
     7 :  6 : BoundaryDiscreteReset :   
     8 :  4 : RayleighDiscreteReset :   
     9 :  3 : AbsorptionDiscreteReset :   
    10 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
    11 :  7 : BoundaryDiDiTransCoeff :   

    12 :  2 : ScintDiscreteReset :   
    13 :  6 : BoundaryDiscreteReset :   
    14 :  4 : RayleighDiscreteReset :   
    15 :  3 : AbsorptionDiscreteReset :   

    16 :  2 : ScintDiscreteReset :   
    17 :  6 : BoundaryDiscreteReset :   
    18 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
    19 :  7 : BoundaryDiDiTransCoeff :   

    20 :  2 : ScintDiscreteReset :   
    21 :  6 : BoundaryDiscreteReset :   
    22 :  4 : RayleighDiscreteReset :   
    23 :  3 : AbsorptionDiscreteReset :   
    24 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :   
    25 :  9 : AbsorptionEffDetect :   
    26 :  0 : Unclassified :   



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



TODO : investigate impact of U4Process::ClearNumberOfInteractionLengthLeft 
-----------------------------------------------------------------------------

Q: U4Process::ClearNumberOfInteractionLengthLeft will inevitably change the simulation because are using 
   different randoms, but does it change the correctness of the simulation ?

A: Assuming just technical change, because the chances of SC/AB etc..
   are surely independent of what happened before ? 

To verify the assumption need high stats statistical comparison of history frequencies 
with and without this trick being applied. 
This will require getting the statistical comparison python machinery into new workflow
using the new SEvt arrays.  


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


* with this the step point preamble now 2,6,4,3 with all 4 process reset for every step point
* the advantage of this is its simplicity and similarity of each step point 

* the preamble consumption can loosely be regarded as the arrows between flag points, 
  that act to decide what the next history flag will be::

  TO->BT->BT->SA 

* where does SA fit into this ? B:G4 is getting NoRINDEX truncated ?
  but A actually finds perfectAbsorbSurface boundary

* DONE: added Geant4 surface equivalent on the Rock///Air boundary  
  which succeeds to avoid the dirty NoRINDEX truncation 


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


    ## After remove the NoRINDEX kludge and add the G4OpticalSurface
    ## get additional tail of 8,9 

    In [2]: bt[:5,:20]
    Out[2]: 
    array([[2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 2, 6, 8, 9, 0, 0, 0, 0, 0, 0],
           [2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 7, 2, 6, 4, 3, 8, 9, 0, 0]], dtype=uint8)


    In [1]: print(stack.label(bt[0,:20]))
     0 :  2 : ScintDiscreteReset :  
     1 :  6 : BoundaryDiscreteReset :  
     2 :  4 : RayleighDiscreteReset :  
     3 :  3 : AbsorptionDiscreteReset :  
     4 :  8 : BoundaryReflectTransmitAbsorb :  
     5 :  7 : BoundaryDiDiTransCoeff : 

     6 :  2 : ScintDiscreteReset :  
     7 :  6 : BoundaryDiscreteReset :  
     8 :  4 : RayleighDiscreteReset :  
     9 :  3 : AbsorptionDiscreteReset :  
    10 :  8 : BoundaryReflectTransmitAbsorb :  
    11 :  7 : BoundaryDiDiTransCoeff :  

    12 :  2 : ScintDiscreteReset :  
    13 :  6 : BoundaryDiscreteReset :  
    14 :  4 : RayleighDiscreteReset :  
    15 :  3 : AbsorptionDiscreteReset :  
    16 :  8 : BoundaryReflectTransmitAbsorb :  
    17 :  9 : AbsorptionEffDetect :  

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

  * DONE : ADD A GEANT4 SURFACE TO THE TEST GEOMETRY TO MAKE THE TAIL POSSIBLE TO ALIGN WITH


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


