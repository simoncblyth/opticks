GROUPVEL
==========

Approach
-----------

This is a longstanding issue of proagation time mismatch between Opticks and G4

Longterm solution is to export GROUPVEL property together with 
RINDEX and others in the G4DAE export.  Not prepared to go there
yet though, so need a shortcut way to get this property into the
Opticks boundary texture.


See Also
---------

* ana/groupvel.py 
* ggeo/GProperty<T>::make_GROUPVEL




DsG4OpBoundaryProcess dumping : looks like getting groupvel from Ac instead of LS and MO
-------------------------------------------------------------------------------------------

tconcentric-i::

    In [2]: ab.b.sel = "TO BT BT BT BT SA"

    In [6]: ab.b.psel_dindex(slice(0,100))     # first 100 of top line, straight thrus (easy to interpret)
    Out[6]: '--dindex=1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,19,20,23,25,27,29,31,35,36,37,38,39,40,41,42,43,47,48,49,50,52,55,58,60,61,67,72,73,74,75,76,78,79,80,82,86,87,89,93,94,95,96,97'


In [1]: ab.b.psel_dindex(limit=10, reverse=True)
Out[1]: '--dindex=999999,999997,999996,999995,999994,999993,999992,999991,999990,999989'




Back to basics after moving to fine domain (1nm)
--------------------------------------------------

::

   tconcentric-tt --finedbndtex


Fine domain means can no longer blame interpolation mismatch for discreps

::
 
                    |
                    | 3000             4000             5000
         0          | + |               +                +
         +          |   |             |   |            |   |
        TO         BT   BT            BT  BT           SA  
              0     | 1 |      2      | 3 |     4      |   |
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 

Calculate expectations for global times with tconcentric geometry, in bnd.py::

    Gd,LS,Ac,MO = 0,1,2,3
    gvel = i1m.data[(Gd,Ac,LS,Ac,MO),1,430-60,0]
    dist = np.array([0,3000-5,3000+5,4000-5,4000+5,5000-5], dtype=np.float32)   # tconcentric radii
    ddif = np.diff(dist)
    tdif = ddif/gvel
    tabs = np.cumsum(ddif/gvel) + 0.1 

    print "gvel: %r " %  gvel
    print "dist: %r " %  dist
    print "ddif: %r " %  ddif
    print "tdif: %r " %  tdif
    print "tabs: %r " %  tabs

    // with correct groupvel material order : (Gd,Ac,LS,Ac,MO)  get the Opticks times

    gvel: array([ 194.5192,  192.7796,  194.5192,  192.7796,  197.1341], dtype=float32) 
    dist: array([    0.,  2995.,  3005.,  3995.,  4005.,  4995.], dtype=float32) 
    ddif: array([ 2995.,    10.,   990.,    10.,   990.], dtype=float32) 
    tdif: array([ 15.3969,   0.0519,   5.0895,   0.0519,   5.022 ], dtype=float32) 
    tabs: array([ 15.4969,  15.5488,  20.6383,  20.6902,  25.7121], dtype=float32) 

    // mangling groupvel material order to : (Gd,LS,Ac,MO,Ac) nearly reproduces the CFG4 times...

    gvel2: array([ 194.5192,  194.5192,  192.7796,  197.1341,  192.7796], dtype=float32) 
    tdif2: array([ 15.3969,   0.0514,   5.1354,   0.0507,   5.1354], dtype=float32) 
    tabs2: array([ 15.4969,  15.5483,  20.6837,  20.7345,  25.8699], dtype=float32) 

    // another mangle to (Gd,LS,Ac,LS,Ac) reproduces the CFG4 times

    gvel3: array([ 194.5192,  194.5192,  192.7796,  194.5192,  192.7796], dtype=float32) 
    tdif3: array([ 15.3969,   0.0514,   5.1354,   0.0514,   5.1354], dtype=float32) 
    tabs3: array([ 15.4969,  15.5483,  20.6837,  20.7352,  25.8706], dtype=float32) 


Hmm looks like difference between use of preVelocity vs postVelocity (are using pre when should be using post).
Potentially due to CRecorder operating PRE_SAVE ?

Hmm to simplify recording, maybe better to move to trajectory style. Collecting steps into a container
within the UserSteppingAction and recording them from the UserTrackingAction after all tracking is done.
See: G4TrackingManager::ProcessOneTrack

::

    202 void G4Trajectory::AppendStep(const G4Step* aStep)
    203 {
    204    positionRecord->push_back( new G4TrajectoryPoint(aStep->GetPostStepPoint()->
    205                                  GetPosition() ));
    206 }
    207 




::

    DsG4OpBoundaryProcess::PostStepDoIt step_id    0 nm        430 priorVelocity    194.519 groupvel_m1            GdDopedLS   194.519 groupvel_m2              Acrylic    192.78 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    1 nm        430 priorVelocity    194.519 groupvel_m1              Acrylic    192.78 groupvel_m2   LiquidScintillator   194.519 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    2 nm        430 priorVelocity     192.78 groupvel_m1   LiquidScintillator   194.519 groupvel_m2              Acrylic    192.78 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    3 nm        430 priorVelocity    194.519 groupvel_m1              Acrylic    192.78 groupvel_m2           MineralOil   197.134 <-proposed 

    // proposed velocity look correct, but suspect the recording happens too soon to feel the effect of it due to PRE_SAVE ??


    CRecorder::RecordStep trackStepLength       2995 trackGlobalTime    15.4969 trackVelocity    194.519 preVelocity    194.519 postVelocity    194.519 preDeltaTime    15.3969 postDeltaTime    15.3969
    CRecorder::RecordStep trackStepLength         10 trackGlobalTime    15.5483 trackVelocity     192.78 preVelocity    194.519 postVelocity     192.78 preDeltaTime  0.0514088 postDeltaTime  0.0518727
    CRecorder::RecordStep trackStepLength        990 trackGlobalTime    20.6837 trackVelocity    194.519 preVelocity     192.78 postVelocity    194.519 preDeltaTime     5.1354 postDeltaTime    5.08947
    CRecorder::RecordStep trackStepLength         10 trackGlobalTime    20.7352 trackVelocity     192.78 preVelocity    194.519 postVelocity     192.78 preDeltaTime  0.0514088 postDeltaTime  0.0518727
    CRecorder::RecordStep trackStepLength        990 trackGlobalTime    25.8706 trackVelocity    197.134 preVelocity     192.78 postVelocity    197.134 preDeltaTime     5.1354 postDeltaTime    5.02196

::
 
     TO   
     BT   Gd/Ac
     BT   Ac/LS
     BT   LS/Ac
     BT   Ac/MO
     SA   MO/Ac





Caution heavy compression with below values::

    ab.sel = "TO BT BT BT BT [SA]"

    a,b = ab.rpost()

    In [42]: a[0]
    Out[42]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136]])

    In [43]: b[0]
    Out[43]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8707]])
    
    In [4]: b[0]   ## after adding BT ProposeVelocity for m2 ... huh why almost no difference 
    Out[4]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4934],
           [ 3004.9551,     0.    ,     0.    ,    15.5458],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8666]])








::

    2016-11-19 14:23:15.001 INFO  [1049278] [CRec::dump@40] CRec::dump record_id 999989 nstp 5  Ori[ 0.0000.0000.000] 
    ( 0)  TO/BT     FrT                                 PRE_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys       GdDopedLS          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns  0.100 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   2995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.497 nm 430.000
     )
    ( 1)  BT/BT     FrT                                            PRE_SAVE 
    [   1](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   2995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.497 nm 430.000
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.548 nm 430.000
     )
    ( 2)  BT/BT     FrT                                            PRE_SAVE 
    [   2](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.548 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.684 nm 430.000
     )
    ( 3)  BT/BT     FrT                                            PRE_SAVE 
    [   3](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.684 nm 430.000
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.735 nm 430.000
     )
    ( 4)  BT/SA     Abs     PRE_SAVE POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   4](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.735 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 25.871 nm 430.000
     )






    
    In [44]: b[0,:,0] == a[0,:,0]    ## 2 simulations yield precisely the same positions
    Out[44]: 
    A()sliced
    A([ True,  True,  True,  True,  True,  True], dtype=bool) 

    In [45]: b[0,:,3] == a[0,:,3]
    Out[45]: 
    A()sliced
    A([ True,  True,  True, False, False, False], dtype=bool)


    In [46]: b[0,:,3] - a[0,:,3]
    Out[46]: 
    A()sliced
    A([ 0.    ,  0.    ,  0.    ,  0.0443,  0.0443,  0.1571])    ## time offset starts in LS, Acrylic does not add to it, MO makes it worse


Group velocity tex props from GdLS,LS,Ac,MO around 430nm::


    In [113]: i1m.data[(0,1,2,3),1,429-60:432-60,0]
    Out[113]: 
    array([[ 194.4354,  194.5192,  194.603 ],
           [ 194.4354,  194.5192,  194.603 ],
           [ 192.6459,  192.7796,  192.9132],
           [ 197.0692,  197.1341,  197.1991]], dtype=float32)

    In [114]: i2m.data[(0,1,2,3),1,429-60:432-60,0]
    Out[114]: 
    array([[ 194.4354,  194.5192,  194.603 ],
           [ 194.4354,  194.5192,  194.603 ],
           [ 192.6459,  192.7796,  192.9132],
           [ 197.0692,  197.1341,  197.1991]], dtype=float32)



Distances, time deltas, velocities for each step::

    In [96]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] )    ## mm
    Out[96]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]))

    In [97]: np.diff( a[0,:,3] ), np.diff( b[0,:,3] )    ## ns 
    Out[97]: 
    A([ 15.3967,       0.0524,   5.0879,       0.0524,   5.0235]),
    A([ 15.3967,       0.0524,   5.1322,       0.0524,   5.1363]))

              ratio of diffs                  ## mm/ns
    A([ 194.5238,  189.5833,   194.5969,   189.5833,   197.0937]),
    A([ 194.5238,  189.5833,  *192.9167*,  189.5833,  *192.7654*]))

    ##   (TO BT)   (BT BT)     (BT BT)     (BT BT)     (BT SA)          

    ##   Gd         Ac           LS          Ac         MO
    ##
    ## Ac precision very limited due to short time,dist and deep compression ??
    ##
    ## CFG4 gvel numbers for LS and MO look wrong ...
    ##      in fact they look like the Ac numbers  
    ##  


::

    GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 finalVelocity     192.78 priorVelocity    194.519 finalVelocity_m1    194.519
    GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 finalVelocity    194.519 priorVelocity    194.519 finalVelocity_m1     192.78
    GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 finalVelocity     192.78 priorVelocity     192.78 finalVelocity_m1    194.519
    GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 finalVelocity    197.134 priorVelocity    194.519 finalVelocity_m1     192.78


Is there an issue with CRecorder recording the times during stepping before fully baked ?








After 1st try at applying GEANT4_BT_GROUPVEL_FIX minimal change, is there a material swap? that happens on DR?:

    In [5]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] ), np.diff( a[0,:,3] ), np.diff( b[0,:,3] ), np.diff( a[0,:,0] )/np.diff( a[0,:,3] ), np.diff( b[0,:,0] )/np.diff( b[0,:,3] )
    Out[5]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 15.3967,   0.0524,   5.0879,   0.0524,   5.0235]),
    A([ 15.3927,   0.0524,   5.1363,   0.0524,   5.1322]),
    A([ 194.5238,  189.5833,  194.5969,  189.5833,  197.0937]),
    A([ 194.5747,  189.5833,  192.7654,  189.5833,  192.9167]))



::

    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 gv    194.519
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 gv    197.134
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 gv    194.519
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 gv    197.134






::

    In [117]: ab.sel = "TO BT BT BT BT [DR] SA"

    In [118]: a,b = ab.rpost()

    In [119]: a.shape, b.shape
    Out[119]: (7540, 7, 4),  (7677, 7, 4)

    In [123]: a[0]
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136],
           [ 2840.6014,  -320.0011,  4096.1664,    49.2437]])

    In [124]: b[0]
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8707],
           [ 3076.4399,  -722.179 , -3868.4234,    48.579 ]])

    In [126]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] ), np.diff( a[0,:,3] ), np.diff( b[0,:,3] ), np.diff( a[0,:,0] )/np.diff( a[0,:,3] ), np.diff( b[0,:,0] )/np.diff( b[0,:,3] )
    Out[126]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 , -2154.4702]),   A.dx mm
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 , -1918.6317]),   B.dx mm
    A([ 15.3967,       0.0524,   5.0879,       0.0524,   5.0235,  23.5301]),       A.dt ns
    A([ 15.3967,       0.0524,   5.1322,       0.0524,   5.1363,  22.7083]),       B.dt ns
    A([ 194.5238,    189.5833,  194.5969,    189.5833,  197.0937,  -91.5622]),     A.gv mm/ns
    A([ 194.5238,    189.5833,  192.9167,    189.5833,  192.7654,  -84.4902]))     B.gv mm/ns

    ## consistent issue, slow LS and MO groupvel in CFG4 (looking like Ac groupvel)





Suspect seeing G4 bug that is fixed in lastest G4 with the below special case GROUPVEL access for



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc::

     165 G4VParticleChange*
     166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     167 {
     ...
     529 
     530         aParticleChange.ProposeMomentumDirection(NewMomentum);
     531         aParticleChange.ProposePolarization(NewPolarization);
     532 
     533         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     534            G4MaterialPropertyVector* groupvel =
     535            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     536            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     537            aParticleChange.ProposeVelocity(finalVelocity);
     538         }
     539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }

Looking for the bug that induced the above special case, yeilds zilch.

* https://bugzilla-geant4.kek.jp/buglist.cgi?component=processes%2Foptical&product=Geant4

Try looking at code history

* http://www-geant4.kek.jp/lxr/source//processes/optical/src/G4OpBoundaryProcess.cc
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=8.0  Not there
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=9.5  Nope
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=9.6  First appearance, for only FresnelRefraction

::

    497         if ( theStatus == FresnelRefraction ) {
    498            G4MaterialPropertyVector* groupvel =
    499            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    500            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    501            aParticleChange.ProposeVelocity(finalVelocity);
    502         }

* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=10.1 Adds in Transmission

::

    532         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
    533            G4MaterialPropertyVector* groupvel =
    534            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    535            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    536            aParticleChange.ProposeVelocity(finalVelocity);
    537         }
    538 

Look for commit history, Geant4 svn is hidden behind CERN login, try mirrors.

The below have no history

* https://gitlab.cern.ch/geant4/geant4/commits/master/source/processes/optical/src/G4OpBoundaryProcess.cc
* https://github.com/alisw/geant4


Add to cfg4/DsG4OpBoundaryProcess.cc::

     600         
     601 #ifdef GEANT4_BT_GROUPVEL_FIX
     602     // from /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc
     603        if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     604            G4MaterialPropertyVector* groupvel =
     605            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     606            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     607            aParticleChange.ProposeVelocity(finalVelocity);
     608         }
     609 #endif  
     610 




::

    112 G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    113                             const G4Track& ,
    114                             const G4Step&
    115                             )
    116 {
    117 //  clear NumberOfInteractionLengthLeft
    118     ClearNumberOfInteractionLengthLeft();
    119 
    120     return pParticleChange;
    121 }






where does G4 set times anyhow
--------------------------------


Recorded time comes from::

     820 void CRecorder::RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ )
     821 {
     822     const G4ThreeVector& pos = point->GetPosition();
     823     const G4ThreeVector& pol = point->GetPolarization();
     824 
     825     G4double time = point->GetGlobalTime();

::

    delta:cfg4 blyth$ g4-cc SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/models/fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(fe->getNeutronAge(i)*second);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/models/fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(fe->getPhotonAge(i)*second);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc:  thePro.SetGlobalTime(0.0);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc:    thePro.SetGlobalTime(0.0);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc:  p->SetGlobalTime(time);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForDecay.cc:  pPostStepPoint->SetGlobalTime( GetGlobalTime() );
    delta:cfg4 blyth$ 
    delta:cfg4 blyth$ g4-hh SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/util/include/G4HadProjectile.hh:  inline void SetGlobalTime(G4double t);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/util/include/G4HadProjectile.hh:inline void G4HadProjectile::SetGlobalTime(G4double t) 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4StepPoint.hh:   void SetGlobalTime(const G4double aValue);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Track.hh:   void SetGlobalTime(const G4double aValue);
    delta:cfg4 blyth$ 
    delta:cfg4 blyth$ g4-icc SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Step.icc:   fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Step.icc:   fpTrack->SetGlobalTime(fpPostStepPoint->GetGlobalTime());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4StepPoint.icc: void G4StepPoint::SetGlobalTime(const G4double aValue)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Track.icc:   inline void G4Track::SetGlobalTime(const G4double aValue)
    delta:cfg4 blyth$ 



::

    (lldb) b "G4StepPoint::SetGlobalTime(const G4double)"


Looks like step point time only ever set at initialization, from the track::

    (lldb) bt
    * thread #1: tid = 0x1059dc, 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(G4Track*) [inlined] G4StepPoint::SetGlobalTime(this=0x000000011127a650, aValue=<unavailable>) at G4StepPoint.icc:60, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(G4Track*) [inlined] G4StepPoint::SetGlobalTime(this=0x000000011127a650, aValue=<unavailable>) at G4StepPoint.icc:60
        frame #1: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(this=0x000000011127a5f0, aValue=0x000000012818a7b0) + 89 at G4Step.icc:200
        frame #2: 0x0000000104c7502c libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x000000011127a460, valueTrack=<unavailable>) + 1468 at G4SteppingManager.cc:356
        frame #3: 0x0000000104c7e4a7 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011127a420, apValueG4Track=<unavailable>) + 199 at G4TrackingManager.cc:89
        frame #4: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011127a390, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #5: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010f66ef00, i_event=0) + 49 at G4RunManager.cc:399
        frame #6: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010f66ef00, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #7: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010f66ef00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #8: 0x0000000103ee4882 libcfg4.dylib`CG4::propagate(this=0x000000010f66ee50) + 1458 at CG4.cc:270
        frame #9: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3f0) + 538 at OKG4Mgr.cc:82
        frame #10: 0x00000001000139ca OKG4Test`main(argc=29, argv=0x00007fff5fbfe4d0) + 1498 at OKG4Test.cc:57
        frame #11: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) f 2
    frame #2: 0x0000000104c7502c libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x000000011127a460, valueTrack=<unavailable>) + 1468 at G4SteppingManager.cc:356
       353     }
       354     else {
       355  // Initial set up for attribues of 'Step'
    -> 356         fStep->InitializeStep( fTrack );
       357     }
       358  #ifdef G4VERBOSE
       359                           // !!!!! Verbose
    (lldb) f 1
    frame #1: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(this=0x000000011127a5f0, aValue=0x000000012818a7b0) + 89 at G4Step.icc:200
       197     // To avoid the circular dependency between G4Track, G4Step
       198     // and G4StepPoint, G4Step has to manage the copy actions.
       199     fpPreStepPoint->SetPosition(fpTrack->GetPosition());
    -> 200     fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
       201     fpPreStepPoint->SetLocalTime(fpTrack->GetLocalTime());
       202     fpPreStepPoint->SetProperTime(fpTrack->GetProperTime());
       203     fpPreStepPoint->SetMomentumDirection(fpTrack->GetMomentumDirection());
    (lldb) 


G4Step 
* postStep -> preStep  ?? at each step

* initialize preStepPoint from the track
* updateTrack from the postStepPoint 

::


      track  --->  preStep
           ^
            \
             \___  postStep



::

    184 inline
    185  void G4Step::InitializeStep( G4Track* aValue )
    186  {
    187    // Initialize G4Step attributes
    188    fStepLength = 0.;
    189    fTotalEnergyDeposit = 0.;
    190    fNonIonizingEnergyDeposit = 0.;
    191    fpTrack = aValue;
    192    fpTrack->SetStepLength(0.);
    193 
    194    nSecondaryByLastStep = 0;
    195 
    196    // Initialize G4StepPoint attributes.
    197    // To avoid the circular dependency between G4Track, G4Step
    198    // and G4StepPoint, G4Step has to manage the copy actions.
    199    fpPreStepPoint->SetPosition(fpTrack->GetPosition());
    200    fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
    201    fpPreStepPoint->SetLocalTime(fpTrack->GetLocalTime());
    202    fpPreStepPoint->SetProperTime(fpTrack->GetProperTime());
    203    fpPreStepPoint->SetMomentumDirection(fpTrack->GetMomentumDirection());
    204    fpPreStepPoint->SetKineticEnergy(fpTrack->GetKineticEnergy());
    205    fpPreStepPoint->SetTouchableHandle(fpTrack->GetTouchableHandle());
    206    fpPreStepPoint->SetMaterial( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetMaterial());
    207    fpPreStepPoint->SetMaterialCutsCouple( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetMaterialCutsCouple());
    208    fpPreStepPoint->SetSensitiveDetector( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetSensitiveDetector());
    209    fpPreStepPoint->SetPolarization(fpTrack->GetPolarization());
    210    fpPreStepPoint->SetSafety(0.);
    211    fpPreStepPoint->SetStepStatus(fUndefined);
    212    fpPreStepPoint->SetProcessDefinedStep(0);
    213    fpPreStepPoint->SetMass(fpTrack->GetDynamicParticle()->GetMass());
    214    fpPreStepPoint->SetCharge(fpTrack->GetDynamicParticle()->GetCharge());
    215    fpPreStepPoint->SetWeight(fpTrack->GetWeight());
    216 
    217    // Set Velocity
    218    //  should be placed after SetMaterial for preStep point
    219     fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
    220 
    221    (*fpPostStepPoint) = (*fpPreStepPoint);
    222  }


::

    224 inline
    225  void G4Step::UpdateTrack( )
    226  {
    227    // To avoid the circular dependency between G4Track, G4Step
    228    // and G4StepPoint, G4Step has to manage the update actions.
    229    //  position, time
    230    fpTrack->SetPosition(fpPostStepPoint->GetPosition());
    231    fpTrack->SetGlobalTime(fpPostStepPoint->GetGlobalTime());
    232    fpTrack->SetLocalTime(fpPostStepPoint->GetLocalTime());
    233    fpTrack->SetProperTime(fpPostStepPoint->GetProperTime());
    234    //  energy, momentum, polarization
    235    fpTrack->SetMomentumDirection(fpPostStepPoint->GetMomentumDirection());
    236    fpTrack->SetKineticEnergy(fpPostStepPoint->GetKineticEnergy());
    237    fpTrack->SetPolarization(fpPostStepPoint->GetPolarization());
    238    //  mass charge
    239    G4DynamicParticle* pParticle = (G4DynamicParticle*)(fpTrack->GetDynamicParticle());
    240    pParticle->SetMass(fpPostStepPoint->GetMass());
    241    pParticle->SetCharge(fpPostStepPoint->GetCharge());
    242    //  step length
    243    fpTrack->SetStepLength(fStepLength);
    244    // NextTouchable is updated
    245    // (G4Track::Touchable points touchable of Pre-StepPoint)
    246    fpTrack->SetNextTouchableHandle(fpPostStepPoint->GetTouchableHandle());
    247    fpTrack->SetWeight(fpPostStepPoint->GetWeight());
    248 
    249 
    250    // set velocity
    251    fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());
    252 }

Transportation time setting based on velocity and step length

Breakpoint here is good for seeing track step by step::

    b G4Transportation::AlongStepDoIt

::

    525 G4VParticleChange* G4Transportation::AlongStepDoIt( const G4Track& track,
    526                                                     const G4Step&  stepData )
    527 {
    528   static G4ThreadLocal G4int noCalls=0;
    529   noCalls++;
    530 
    531   fParticleChange.Initialize(track) ;
    532 
    533   //  Code for specific process 
    534   //
    535   fParticleChange.ProposePosition(fTransportEndPosition) ;
    536   fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir) ;
    537   fParticleChange.ProposeEnergy(fTransportEndKineticEnergy) ;
    538   fParticleChange.SetMomentumChanged(fMomentumChanged) ;
    539 
    540   fParticleChange.ProposePolarization(fTransportEndSpin);
    541  
    542   G4double deltaTime = 0.0 ;
    543 
    544   // Calculate  Lab Time of Flight (ONLY if field Equations used it!)
    545   // G4double endTime   = fCandidateEndGlobalTime;
    546   // G4double delta_time = endTime - startTime;
    547 
    548   G4double startTime = track.GetGlobalTime() ;
    549  
    550   if (!fEndGlobalTimeComputed)
    ////
    ////    fEndGlobalTimeComputed always false without magnetic field ???
    ////    THIS LOOKS TO BE WHERE THE TIMES ARE COMING FROM
    ////    USING prestep point velocity and steplength
    ////
    551   {
    552      // The time was not integrated .. make the best estimate possible
    553      //
    554      G4double initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
    555      G4double stepLength      = track.GetStepLength();
    556 
    557      deltaTime= 0.0;  // in case initialVelocity = 0 
    558      if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
    559 
    560      fCandidateEndGlobalTime   = startTime + deltaTime ;
    561      fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
    562   }
    563   else
    564   {
    565      deltaTime = fCandidateEndGlobalTime - startTime ;
    566      fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    567   }
    568 
    569 
    570   // Now Correct by Lorentz factor to get delta "proper" Time
    571  
    572   G4double  restMass       = track.GetDynamicParticle()->GetMass() ;
    573   G4double deltaProperTime = deltaTime*( restMass/track.GetTotalEnergy() ) ;


::

    (lldb) p track
    (const G4Track) $14 = {
      fCurrentStepNumber = 1
      fPosition = (dx = 0, dy = 0, dz = 0)
      fGlobalTime = 0.10000000149011612
      fLocalTime = 0
      fTrackLength = 0
      fParentID = 0
      fTrackID = 9999
      fVelocity = 299.79245800000001
      fpTouchable = {
        fObj = 0x000000012788ed70
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed70
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 2995
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000110104b20
      groupvel = 0x0000000110105760
      prev_velocity = 194.51919555664063
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }
    (lldb) p fTransportEndPosition
    (G4ThreeVector) $15 = (dx = 2995, dy = 0, dz = 0)
    (lldb) 


    (lldb) p *stepData.GetPreStepPoint()
    (G4StepPoint) $19 = {
      fPosition = (dx = 0, dy = 0, dz = 0)
      fGlobalTime = 0.10000000149011612
      fLocalTime = 0
      fProperTime = 0
      fMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fKineticEnergy = 0.0000028833531986511571
      fVelocity = 194.51919555664063
      fpTouchable = {
        fObj = 0x000000012788ed70
      }
      fpMaterial = 0x0000000110104b20
      fpMaterialCutsCouple = 0x0000000110609570
      fpSensitiveDetector = 0x0000000000000000
      fSafety = 0
      fPolarization = (dx = 0, dy = 1, dz = 0)
      fStepStatus = fUndefined
      fpProcessDefinedStep = 0x0000000000000000
      fMass = 0
      fCharge = 0
      fMagneticMoment = 0
      fWeight = 1
    }

    (lldb) p fCandidateEndGlobalTime 
    (G4double) $20 = 15.49693803338686


    (lldb) p 2995./194.51919555664063
    (double) $21 = 15.396938031896743
    (lldb) p 2995./194.51919555664063 + 0.10000000149011612
    (double) $22 = 15.49693803338686
    (lldb) 


    (lldb) p track
    (const G4Track) $23 = {
      fCurrentStepNumber = 2
      fPosition = (dx = 2995, dy = 0, dz = 0)
      fGlobalTime = 15.49693803338686
      fLocalTime = 15.396938031896743
      fTrackLength = 2995     
      /// not including current step

      fParentID = 0
      fTrackID = 9999
      fVelocity = 194.51919555664063

      /// this velocity is not for Acrylic ??
      ///
      fpTouchable = {
        fObj = 0x000000012788ed80
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed80
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 10
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000110104b20
      groupvel = 0x0000000110105760
      prev_velocity = 194.51919555664063
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }


::

    (ldb) p *stepData.GetPreStepPoint()
    (G4StepPoint) $24 = {
      fPosition = (dx = 2995, dy = 0, dz = 0)
      fGlobalTime = 15.49693803338686
      fLocalTime = 15.396938031896743
      fProperTime = 0
      fMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fKineticEnergy = 0.0000028833531986511571
      fVelocity = 194.51919555664063
      fpTouchable = {
        fObj = 0x000000012788ed80
      }
      fpMaterial = 0x00000001100f93d0
      fpMaterialCutsCouple = 0x0000000110608660
      fpSensitiveDetector = 0x0000000000000000
      fSafety = 0.00000000050000000000000003
      fPolarization = (dx = 0, dy = 1, dz = 0)
      fStepStatus = fGeomBoundary
      fpProcessDefinedStep = 0x000000011011f4b0
      fMass = 0
      fCharge = 0
      fMagneticMoment = 0
      fWeight = 1
    }


    (lldb) p stepData.GetPreStepPoint()->GetMaterial()->GetName()
    (const G4String) $26 = (std::__1::string = "Acrylic")
    (lldb) p stepData.GetPostStepPoint()->GetMaterial()->GetName()
    (const G4String) $27 = (std::__1::string = "Acrylic")

    (lldb) p track
    (const G4Track) $28 = {
      fCurrentStepNumber = 3
      fPosition = (dx = 3005, dy = 0, dz = 0)
      fGlobalTime = 15.548346841506715
      fLocalTime = 15.448346840016599
      fTrackLength = 3005
      fParentID = 0
      fTrackID = 9999
      fVelocity = 192.77955627441406
      fpTouchable = {
        fObj = 0x000000012788ed90
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed90
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 990
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x00000001100f93d0
      groupvel = 0x00000001101004f0
      prev_velocity = 192.77955627441406
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }



::

    (lldb) b "G4Track::SetGlobalTime"

    (lldb) bt
    * thread #1: tid = 0x1059dc, 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Track::SetGlobalTime(this=0x000000012818a7b0, aValue=<unavailable>) at G4Track.icc:100, queue = 'com.apple.main-thread', stop reason = breakpoint 4.3
      * frame #0: 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Track::SetGlobalTime(this=0x000000012818a7b0, aValue=<unavailable>) at G4Track.icc:100
        frame #1: 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Step::UpdateTrack(this=0x000000011127a5f0) + 34 at G4Step.icc:231
        frame #2: 0x0000000104c76b3e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000011127a460) + 510 at G4SteppingManager2.cc:471
        frame #3: 0x0000000104c74771 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000011127a460) + 417 at G4SteppingManager.cc:191
        frame #4: 0x0000000104c7e771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011127a420, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #5: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011127a390, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #6: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010f66ef00, i_event=0) + 49 at G4RunManager.cc:399
        frame #7: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010f66ef00, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #8: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010f66ef00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #9: 0x0000000103ee4882 libcfg4.dylib`CG4::propagate(this=0x000000010f66ee50) + 1458 at CG4.cc:270
        frame #10: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3f0) + 538 at OKG4Mgr.cc:82
        frame #11: 0x00000001000139ca OKG4Test`main(argc=29, argv=0x00007fff5fbfe4d0) + 1498 at OKG4Test.cc:57
        frame #12: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) 
















::

     49  G4double G4ParticleChange::GetVelocity() const
     50 {
     51    return theVelocityChange;
     52 }
     53 
     54 inline
     55   void G4ParticleChange::ProposeVelocity(G4double finalVelocity)
     56 {
     57    theVelocityChange = finalVelocity;
     58    isVelocityChanged = true;
     59 }
     60 

::

    228 void G4ParticleChange::Initialize(const G4Track& track)
    229 {
    230   // use base class's method at first
    231   G4VParticleChange::Initialize(track);
    232   theCurrentTrack= &track;
    233 
    234   // set Energy/Momentum etc. equal to those of the parent particle
    235   const G4DynamicParticle*  pParticle = track.GetDynamicParticle();
    236   theEnergyChange            = pParticle->GetKineticEnergy();
    237   theVelocityChange          = track.GetVelocity();
    238   isVelocityChanged          = false;
    239   theMomentumDirectionChange = pParticle->GetMomentumDirection();
    240   thePolarizationChange      = pParticle->GetPolarization();
    241   theProperTimeChange        = pParticle->GetProperTime();
    242 
    243   // Set mass/charge/MagneticMoment  of DynamicParticle
    244   theMassChange = pParticle->GetMass();
    245   theChargeChange = pParticle->GetCharge();
    246   theMagneticMomentChange = pParticle->GetMagneticMoment();
    247 
    248   // set Position  equal to those of the parent track
    249   thePositionChange      = track.GetPosition();
    250 
    251   // set TimeChange equal to local time of the parent track
    252   theTimeChange                = track.GetLocalTime();
    253 
    254   // set initial Local/Global time of the parent track
    255   theLocalTime0           = track.GetLocalTime();
    256   theGlobalTime0          = track.GetGlobalTime();
    257 
    258 }


::

    348 G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    349 {
    350   // A physics process always calculates the final state of the particle
    351 
    352   // Take note that the return type of GetMomentumChange is a
    353   // pointer to G4ParticleMometum. Also it is a normalized 
    354   // momentum vector.
    355 
    356   G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint();
    357   G4Track* pTrack = pStep->GetTrack();
    358 
    359   // Set Mass/Charge
    360   pPostStepPoint->SetMass(theMassChange);
    361   pPostStepPoint->SetCharge(theChargeChange);
    362   pPostStepPoint->SetMagneticMoment(theMagneticMomentChange);
    363 
    364   // update kinetic energy and momentum direction
    365   pPostStepPoint->SetMomentumDirection(theMomentumDirectionChange);
    366   pPostStepPoint->SetKineticEnergy( theEnergyChange );
    367 
    368   // calculate velocity
    369   pTrack->SetKineticEnergy( theEnergyChange );
    370   if (!isVelocityChanged) {
    371     if(theEnergyChange > 0.0) {
    372       theVelocityChange = pTrack->CalculateVelocity();
    373     } else if(theMassChange > 0.0) {
    374       theVelocityChange = 0.0;
    375     }
    376   }
    377   pPostStepPoint->SetVelocity(theVelocityChange);

    ///   the G4ParticleChange::GetVelocity is never called
    ///   so passing on to post is the only place the info
    ///   goes


    378 
    379    // update polarization
    380   pPostStepPoint->SetPolarization( thePolarizationChange );
    381 
    382   // update position and time
    383   pPostStepPoint->SetPosition( thePositionChange  );
    384   pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
    385   pPostStepPoint->SetLocalTime( theTimeChange );
    386   pPostStepPoint->SetProperTime( theProperTimeChange  );
    387 
    388   if (isParentWeightProposed ){
    389     pPostStepPoint->SetWeight( theParentWeight );
    390   }
    391 
    392 #ifdef G4VERBOSE
    393   G4Track*     aTrack  = pStep->GetTrack();
    394   if (debugFlag) CheckIt(*aTrack);
    395 #endif
    396 
    397   //  Update the G4Step specific attributes 
    398   return UpdateStepInfo(pStep);
    399 }


::

    delta:cfg4 blyth$ g4-cc UpdateStepForPostStep
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/generic/src/G4ParticleChangeForOccurenceBiasing.cc:G4Step* G4ParticleChangeForOccurenceBiasing::UpdateStepForPostStep(G4Step* step)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/generic/src/G4ParticleChangeForOccurenceBiasing.cc:  fWrappedParticleChange->UpdateStepForPostStep(step);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITReactionChange.cc:  fParticleChange[stepA->GetTrack()]->UpdateStepForPostStep(stepA);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITReactionChange.cc:  fParticleChange[stepB->GetTrack()]->UpdateStepForPostStep(stepB);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpParticleChange->UpdateStepForPostStep(fpStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:G4Step* G4FastStep::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChange.cc:G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForDecay.cc:G4Step* G4ParticleChangeForDecay::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForGamma.cc:G4Step* G4ParticleChangeForGamma::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForLoss.cc:G4Step* G4ParticleChangeForLoss::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForMSC.cc:G4Step* G4ParticleChangeForMSC::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForTransport.cc:G4Step* G4ParticleChangeForTransport::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForTransport.cc:  // return G4ParticleChange::UpdateStepForPostStep(pStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4VParticleChange.cc:G4Step* G4VParticleChange::UpdateStepForPostStep(G4Step* Step)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager2.cc:  fParticleChange->UpdateStepForPostStep(fStep);
    delta:cfg4 blyth$ 

::

    526 void G4SteppingManager::InvokePSDIP(size_t np)
    527 {
    528          fCurrentProcess = (*fPostStepDoItVector)[np];
    529          fParticleChange
    530             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
    531 
    532          // Update PostStepPoint of Step according to ParticleChange
    533      fParticleChange->UpdateStepForPostStep(fStep);
    ...
    538          // Update G4Track according to ParticleChange after each PostStepDoIt
    539          fStep->UpdateTrack();
    540 
    541          // Update safety after each invocation of PostStepDoIts
    542          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    543 
    544          // Now Store the secondaries from ParticleChange to SecondaryList
    545          G4Track* tempSecondaryTrack;
    546          G4int    num2ndaries;
    547 
    548          num2ndaries = fParticleChange->GetNumberOfSecondaries();
    ...
    ...      skipped 2ndary loop
    ...
    581          // Set the track status according to what the process defined
    582          fTrack->SetTrackStatus( fParticleChange->GetTrackStatus() );
    ...
    585          fParticleChange->Clear();
    586 }



::

    116 G4StepStatus G4SteppingManager::Stepping()
    117 //////////////////////////////////////////
    118 {
    ...
    133 
    134 // Store last PostStepPoint to PreStepPoint, and swap current and nex
    135 // volume information of G4Track. Reset total energy deposit in one Step. 
    136    fStep->CopyPostToPreStepPoint();
    137    fStep->ResetTotalEnergyDeposit();
    138 
    139 // Switch next touchable in track to current one
    140    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
    ...
    147 //JA Set the volume before it is used (in DefineStepLength() for User Limit) 
    148    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    149 
    150 // Reset the step's auxiliary points vector pointer
    151    fStep->SetPointerToVectorOfAuxiliaryPoints(0);
    152 
    153 //-----------------
    154 // AtRest Processes
    155 //-----------------
    156 
    157    if( fTrack->GetTrackStatus() == fStopButAlive ){
    158      if( MAXofAtRestLoops>0 ){
    159         InvokeAtRestDoItProcs();
    160         fStepStatus = fAtRestDoItProc;
    161         fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    162 
    163 #ifdef G4VERBOSE
    164             // !!!!! Verbose
    165              if(verboseLevel>0) fVerbose->AtRestDoItInvoked();
    166 #endif
    167 
    168      }
    169      // Make sure the track is killed
    170      fTrack->SetTrackStatus( fStopAndKill );
    171    }
    172
    173 //---------------------------------
    174 // AlongStep and PostStep Processes
    175 //---------------------------------
    176 
    177 
    178    else{
    179      // Find minimum Step length demanded by active disc./cont. processes
    180      DefinePhysicalStepLength();
    181 
    182      // Store the Step length (geometrical length) to G4Step and G4Track
    183      fStep->SetStepLength( PhysicalStep );
    184      fTrack->SetStepLength( PhysicalStep );
    185      G4double GeomStepLength = PhysicalStep;
    186 
    187      // Store StepStatus to PostStepPoint
    188      fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    189 
    190      // Invoke AlongStepDoIt 
    191      InvokeAlongStepDoItProcs();
    192 
    193      // Update track by taking into account all changes by AlongStepDoIt
    194      fStep->UpdateTrack();
    195 
    196      // Update safety after invocation of all AlongStepDoIts
    197      endpointSafOrigin= fPostStepPoint->GetPosition();
    198 //     endpointSafety=  std::max( proposedSafety - GeomStepLength, 0.);
    199      endpointSafety=  std::max( proposedSafety - GeomStepLength, kCarTolerance);
    200 
    201      fStep->GetPostStepPoint()->SetSafety( endpointSafety );
    202 
    203 #ifdef G4VERBOSE
    204                          // !!!!! Verbose
    205            if(verboseLevel>0) fVerbose->AlongStepDoItAllDone();
    206 #endif
    207 
    208      // Invoke PostStepDoIt
    209      InvokePostStepDoItProcs();
    210 
    211 #ifdef G4VERBOSE
    212                  // !!!!! Verbose
    213      if(verboseLevel>0) fVerbose->PostStepDoItAllDone();
    214 #endif
    215    }














tconcentric check
--------------------

::

    In [2]: ab.sel = "TO BT BT BT BT SA"    ## straight thru selection

    In [3]: a,b = ab.rpost()

    In [4]: a.shape
    Out[4]: (669843, 6, 4)

    In [5]: b.shape
    Out[5]: (671267, 6, 4)

    In [7]: a[0]    ## positions match, times off a little
    Out[7]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136]])

    In [8]: b[0]
    Out[8]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4934],
           [ 3004.9551,     0.    ,     0.    ,    15.5458],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8666]])


    In [35]: np.diff(a[0,:,0])/np.diff(a[0,:,3])  ## ratio of x diff to t diff -> groupvel in Gd Ac LS Ac MO for  429.5686 nm
    A([ 194.5238,  189.5833,  194.5969,  189.5833,  197.0937])

    In [36]: np.diff(b[0,:,0])/np.diff(b[0,:,3])
    A([ 194.5747,  189.5833,  192.7654,  189.5833,  192.9167])

    In [13]: np.diff(a[0,:,0])/np.diff(a[0,:,3]) - np.diff(b[0,:,0])/np.diff(b[0,:,3])
    A([-0.0509,  0.    ,  1.8315,  0.    ,  4.177 ])    ## mm/ns

    ## fairly close, possibly can attribute to interpolation differences ???


Review
--------

* http://www.hep.man.ac.uk/u/roger/PHYS10302/lecture15.pdf
* http://web.ift.uib.no/AMOS/PHYS261/opticsPDF/Examples_solutions_phys263.pdf

::
                
    .
          c          w  dn           c           
    vg = --- (  1 +  -- ---  )   ~  --- (  1 +   ?  )
          n          n  dw           n              


     d logn      dn   1  
     ------ =   ---  --- 
      dw         dw   n


     d logw      dw   1             dn/n       dn   w
     ------ =   ---  ---    ->     -----  =    ---  -
      dn         dn   w            d logw       dw   n


     c          dn / n 
    --- ( 1 +   ---    )
     n          d logw


     c          dn  
     -   +   c  ---
     n          dlogw




                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    G4 using this energy domain approach approximating the dispersion part E dn/dE as shown below

                c                  n1 - n0         n1 - n0               dn        dn    dE          
    vg =  -----------       ds = ------------  =  ------------     ~   ------  =  ---- ------- =  E dn/dE 
           nn +  ds               log(E1/E0)      log E1 - log E0      d(logE)     dE   dlogE        
  



Now get G4 warnings when run without groupvel option
-------------------------------------------------------

::

    634   accuracy = theVelocityChange/c_light - 1.0;
    635   if (accuracy > accuracyForWarning) {
    636     itsOKforVelocity = false;
    637     nError += 1;
    638     exitWithError = exitWithError ||  (accuracy > accuracyForException);
    639 #ifdef G4VERBOSE
    640     if (nError < maxError) {
    641       G4cout << "  G4ParticleChange::CheckIt    : ";
    642       G4cout << "the velocity is greater than c_light  !!" << G4endl;
    643       G4cout << "  Velocity:  " << theVelocityChange/c_light  <<G4endl;
    644       G4cout << aTrack.GetDefinition()->GetParticleName()
    645          << " E=" << aTrack.GetKineticEnergy()/MeV
    646          << " pos=" << aTrack.GetPosition().x()/m
    647          << ", " << aTrack.GetPosition().y()/m
    648          << ", " << aTrack.GetPosition().z()/m
    649          <<G4endl;
    650     }
    651 #endif
    652   }



    2016-11-10 17:03:42.091 INFO  [373895] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                3e+03
            Stepping Control      :                    0
        First Step In the voulme  : 
        Last Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :             1.19e+03
            Position - y (mm)   :                 -130
            Position - z (mm)   :             2.75e+03
            Time (ns)           :                 9.98
            Proper Time (ns)    :                    0
            Momentum Direct - x :                0.397
            Momentum Direct - y :              -0.0435
            Momentum Direct - z :                0.917
            Kinetic Energy (MeV):             2.88e-06
            Velocity  (/c):                    1
            Polarization - x    :                0.918
            Polarization - y    :               0.0188
            Polarization - z    :               -0.396
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------

::

    254 ///////////////////
    255 G4double G4Track::CalculateVelocityForOpticalPhoton() const
    256 ///////////////////
    257 {
    258    
    259   G4double velocity = c_light ;
    260  
    261 
    262   G4Material* mat=0;
    263   G4bool update_groupvel = false;
    264   if ( fpStep !=0  ){
    265     mat= this->GetMaterial();         //   Fix for repeated volumes
    266   }else{
    267     if (fpTouchable!=0){
    268       mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    269     }
    270   }
    271   // check if previous step is in the same volume
    272     //  and get new GROUPVELOCITY table if necessary 
    273   if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
    274     groupvel = 0;
    275     if(mat->GetMaterialPropertiesTable() != 0)
    276       groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    277     update_groupvel = true;
    278   }
    279   prev_mat = mat;
    280  
    281   if  (groupvel != 0 ) {
    282     // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
    283     // values stored in GROUPVEL material properties vector
    284     velocity =  prev_velocity;
    285    
    286     // check if momentum is same as in the previous step
    287     //  and calculate group velocity if necessary 
    288     G4double current_momentum = fpDynamicParticle->GetTotalMomentum();
    289     if( update_groupvel || (current_momentum != prev_momentum) ) {
    290       velocity =
    291     groupvel->Value(current_momentum);
    292       prev_velocity = velocity;
    293       prev_momentum = current_momentum;
    294     }
    295   }  
    296  
    297   return velocity ;
    298 }







Opticks GROUPVEL
------------------

::

    simon:cfg4 blyth$ opticks-find GROUPVEL 
    ./cfg4/CPropLib.cc: GROUPVEL kludge causing "generational" confusion
    ./cfg4/CPropLib.cc:             LOG(info) << "CPropLib::makeMaterialPropertiesTable applying GROUPVEL kludge" ; 
    ./cfg4/CPropLib.cc:             addProperty(mpt, "GROUPVEL", prop );
    ./cfg4/CPropLib.cc:    bool groupvel = strcmp(lkey, "GROUPVEL") == 0 ; 
    ./cfg4/CTraverser.cc:const char* CTraverser::GROUPVEL = "GROUPVEL" ; 
    ./cfg4/CTraverser.cc:    // First get of GROUPVEL property creates it 
    ./cfg4/CTraverser.cc:            G4MaterialPropertyVector* gv = mpt->GetProperty(GROUPVEL);  
    ./cfg4/tests/CInterpolationTest.cc:    const char* mkeys_1 = "GROUPVEL,,," ;
    ./ggeo/GGeoTestConfig.cc:const char* GGeoTestConfig::GROUPVEL_ = "groupvel"; 
    ./ggeo/GGeoTestConfig.cc:    else if(strcmp(k,GROUPVEL_)==0)   arg = GROUPVEL ; 
    ./ggeo/GGeoTestConfig.cc:        case GROUPVEL       : setGroupvel(s)       ;break;
    ./ggeo/GMaterialLib.cc:"group_velocity:GROUPVEL,"
    ./cfg4/CTraverser.hh:        static const char* GROUPVEL ; 
    ./ggeo/GGeoTestConfig.hh:                      GROUPVEL,
    ./ggeo/GGeoTestConfig.hh:       static const char* GROUPVEL_ ; 
    simon:opticks blyth$ 



G4 GROUPVEL
--------------

::

    simon:geant4_10_02_p01 blyth$ find source -name '*.*' -exec grep -H GROUPVEL {} \;
    source/materials/include/G4MaterialPropertiesTable.hh:// Updated:     2005-05-12 add SetGROUPVEL() by P. Gumplinger
    source/materials/include/G4MaterialPropertiesTable.hh:    G4MaterialPropertyVector* SetGROUPVEL();
    source/materials/include/G4MaterialPropertiesTable.icc:  //2- So we have a data race if two threads access the same element (GROUPVEL)
    source/materials/include/G4MaterialPropertiesTable.icc:  //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    source/materials/include/G4MaterialPropertiesTable.icc:  //3- SetGROUPVEL() is protected by a mutex that ensures that only
    source/materials/include/G4MaterialPropertiesTable.icc:  //   the same problematic key (GROUPVEL) the mutex will be used.
    source/materials/include/G4MaterialPropertiesTable.icc:  //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    source/materials/include/G4MaterialPropertiesTable.icc:  //   if the map has GROUPVEL key, if so returns immediately.
    source/materials/include/G4MaterialPropertiesTable.icc:  //   group velocity only once even if two threads enter SetGROUPVEL together
    source/materials/include/G4MaterialPropertiesTable.icc:  if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    source/materials/src/G4MaterialPropertiesTable.cc:// Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
    source/materials/src/G4MaterialPropertiesTable.cc:G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    source/materials/src/G4MaterialPropertiesTable.cc:  // check if "GROUPVEL" already exists
    source/materials/src/G4MaterialPropertiesTable.cc:  itr = MPT.find("GROUPVEL");
    source/materials/src/G4MaterialPropertiesTable.cc:  // add GROUPVEL vector
    source/materials/src/G4MaterialPropertiesTable.cc:  // fill GROUPVEL vector using RINDEX values
    source/materials/src/G4MaterialPropertiesTable.cc:    G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:      G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:        G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:  this->AddProperty( "GROUPVEL", groupvel );
    source/processes/optical/src/G4OpBoundaryProcess.cc:           Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    //  and get new GROUPVELOCITY table if necessary 
    source/track/src/G4Track.cc:      groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    // values stored in GROUPVEL material properties vector
    simon:geant4_10_02_p01 blyth$ 




G4Track.cc::

    ///
    ///  GROUPVEL  material property lookup just like RINDEX
    ///            the peculiarity is that the property is 
    ///            derived from RINDEX at first access by special casing in GetProperty
    ///

    317    // cached values for CalculateVelocity  
    318    mutable G4Material*               prev_mat;
    319    mutable G4MaterialPropertyVector* groupvel;
    320    mutable G4double                  prev_velocity;
    321    mutable G4double                  prev_momentum;
    322 


    254 ///////////////////
    255 G4double G4Track::CalculateVelocityForOpticalPhoton() const
    256 ///////////////////
    257 {
    258 
    259   G4double velocity = c_light ;
    260 
    261 
    262   G4Material* mat=0;
    263   G4bool update_groupvel = false;
    264   if ( fpStep !=0  ){
    265     mat= this->GetMaterial();         //   Fix for repeated volumes
    266   }else{
    267     if (fpTouchable!=0){
    268       mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    269     }
    270   }
    271   // check if previous step is in the same volume
    272     //  and get new GROUPVELOCITY table if necessary 
    273   if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
    274     groupvel = 0;
    275     if(mat->GetMaterialPropertiesTable() != 0)
    276       groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    277     update_groupvel = true;
    278   }
    279   prev_mat = mat;
    280 
    281   if  (groupvel != 0 ) {
    282     // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
    283     // values stored in GROUPVEL material properties vector
    284     velocity =  prev_velocity;
    285 
    286     // check if momentum is same as in the previous step
    287     //  and calculate group velocity if necessary 
    288     G4double current_momentum = fpDynamicParticle->GetTotalMomentum();
    289     if( update_groupvel || (current_momentum != prev_momentum) ) {
    290       velocity =
    291     groupvel->Value(current_momentum);
    292       prev_velocity = velocity;
    293       prev_momentum = current_momentum;
    294     }
    295   }
    296 
    297   return velocity ;
    298 }



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc::

     529 
     530         aParticleChange.ProposeMomentumDirection(NewMomentum);
     531         aParticleChange.ProposePolarization(NewPolarization);
     532 
     533         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     534            G4MaterialPropertyVector* groupvel =
     535            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     536            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     537            aParticleChange.ProposeVelocity(finalVelocity);
     538         }
     ///
     ///     such velocity setting not in DsG4OpBoundaryProcess
     ///
     539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }
     544 
     545 void G4OpBoundaryProcess::BoundaryProcessVerbose() const
     546 {




source/materials/include/G4MaterialPropertiesTable.icc::

    115 inline G4MaterialPropertyVector*
    116 G4MaterialPropertiesTable::GetProperty(const char *key)
    117 {
    118   // Returns a Material Property Vector corresponding to a key
    119 
    120   //Important Note for MT. adotti 17 Feb 2016
    121   //In previous implementation the following line was at the bottom of the
    122   //function causing a rare race-condition.
    123   //Moving this line here from the bottom solves the problem because:
    124   //1- Map is accessed only via operator[] (to insert) and find() (to search),
    125   //   and these are thread safe if done on separate elements.
    126   //   See notes on data-races at:
    127   //   http://www.cplusplus.com/reference/map/map/operator%5B%5D/
    128   //   http://www.cplusplus.com/reference/map/map/find/
    129   //2- So we have a data race if two threads access the same element (GROUPVEL)
    130   //   one in read and one in write mode. This was happening with the line
    131   //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    132   //3- SetGROUPVEL() is protected by a mutex that ensures that only
    133   //   one thread at the time will execute its code
    134   //4- The if() statement guarantees that only if two threads are searching
    135   //   the same problematic key (GROUPVEL) the mutex will be used.
    136   //   Different keys do not lock (good for performances)
    137   //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    138   //   if the map has GROUPVEL key, if so returns immediately.
    139   //   This "double check" allows to execute the heavy code to calculate
    140   //   group velocity only once even if two threads enter SetGROUPVEL together
    141   if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    142 
    143   MPTiterator i;
    144   i = MPT.find(G4String(key));
    145   if ( i != MPT.end() ) return i->second;
    146   return NULL;
    147 }

    /// computing a GROUPVEL property vector at first access cause lots of hassle, 
    /// given that RINDEX is constant, should just up front compute GROUPVEL for 
    /// all materials before any event handling happens




::

    119 G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    120 {
    ...
    141   G4MaterialPropertyVector* groupvel = new G4MaterialPropertyVector();
    142 
    146   G4double E0 = rindex->Energy(0);
    147   G4double n0 = (*rindex)[0];
    154   
    160   G4double E1 = rindex->Energy(1);
    161   G4double n1 = (*rindex)[1];
    168 
    169   G4double vg;
    173   vg = c_light/(n0+(n1-n0)/std::log(E1/E0));
    174 
          //   before the loop
          //            E0 = Energy(0)   E1 = Energy(1)      Energy(0) n[0]
          //

    177   if((vg<0) || (vg>c_light/n0))  { vg = c_light/n0; }
    178 
    179   groupvel->InsertValues( E0, vg );
    180 
    184   for (size_t i = 2; i < rindex->GetVectorLength(); i++)
    185   {
    186        vg = c_light/( 0.5*(n0+n1)+(n1-n0)/std::log(E1/E0));

            /// 
            /// note the sleight of hand the same (n1-n0)/std::log(E1/E0) is used for 1st 2 values
            ///

    187 
    190        if((vg<0) || (vg>c_light/(0.5*(n0+n1))))  { vg = c_light/(0.5*(n0+n1)); }

              // at this point in the loop
              //
              // i = 2,    E0 = Energy(0) E1 = Energy(1)    (Energy(0)+Energy(1))/2   // 1st pass using pre-loop settings
              // i = 3,    E0 = Energy(1) E1 = Energy(2)    (Energy(1)+Energy(2))/2   // 2nd pass E0,n0,E1,n1 shunted   
              // i = 4,    E0 = Energy(2) E1 = Energy(3)    (Energy(2)+Energy(3))/2   // 3rd pass E0,n0,E1,n1 shunted   
              //  ....
              // i = N-1   E0 = Energy(N-3)  E1 = Energy(N-2)   (Energy(N-3)+Energy(N-2))/2  


    191        groupvel->InsertValues( 0.5*(E0+E1), vg );
    195        E0 = E1;
    196        n0 = n1;
    197        E1 = rindex->Energy(i);
    198        n1 = (*rindex)[i];
    205   }
    ///
    ///       after the loop 
    ///       "i = N"      E0 = Energy(N-2)   E1 = Energy(N-1)         Energy(N-1)
    ///
    ///     hmmm a difference of bins is needed, but in order not to loose a bin
    ///     a tricky manoever is used of using the 1st and last bin and 
    ///     the average of the body bins
    ///     which means the first bin is half width, and last is 1.5 width
    ///
    ///         0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
    ///         |    /     /     /     /      |
    ///         |   /     /     /     /       |
    ///         0  1     2     3     4        5        <--- still 6 
    ///
    ///  
    ///
    206 
    209   vg = c_light/(n1+(n1-n0)/std::log(E1/E0));
    213   if((vg<0) || (vg>c_light/n1))  { vg = c_light/n1; }
    214   groupvel->InsertValues( E1, vg );
    ... 
    220   
    221   this->AddProperty( "GROUPVEL", groupvel );
    222   
    223   return groupvel;
    224 }

    ///
    ///           Argh... my domain checking cannot to be working...
    ///           this is sticking values midway in energy 
    ///
    ///           Opticks material texture requires fixed domain raster... 
    ///           so either interpolate to get that or adjust the calc ???
    ///


::

   ml = np.load("GMaterialLib.npy")
   wl = np.linspace(60,820,39)
   ri = ml[0,0,:,0]

   c_light = 299.792

   w0 = wl[:-1]
   w1 = wl[1:]

   n0 = ri[:-1]
   n1 = ri[1:]

    In [41]: c_light/(n0 + (n1-n0)/np.log(w1/w0))    # douple flip for e to w, one for reciprocal, one for order ???
    Out[41]: 
    array([ 206.2411,  206.2411,  206.2411,  106.2719,  114.2525, -652.0324,  125.2658,  210.3417,  215.9234,  221.809 ,  228.0242,  234.5973,  207.5104,  209.0361,  210.5849,  212.1565,  213.7514,
            207.991 ,  206.1923,  205.4333,  205.883 ,  206.8385,  207.5627,  208.0809,  206.0739,  205.295 ,  205.4116,  205.5404,  205.7735,  206.0065,  206.2412,  205.3909,  204.2895,  204.3864,
            204.4841,  204.5806,  204.6679,  202.8225])









