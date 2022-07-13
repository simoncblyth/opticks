ab_full_geom_extra_BT : Looks like microStep skipping from CRecorder must be brought to U4Recorder with random rewind
========================================================================================================================

* :doc:`ab_full_geom`


issue 1 : extra points in B : overview
--------------------------------------------

* recall similar issue from degenerate geometry previously, find how that was handled 
* are degenerate skips being applied to A already ?
  If so perhaps those need to be paired with B side handling 
  to keep the bookkeeping the same for microStep 

* what are implication for random alignment when microStep skipping : probably need to 
  rewind the random stream to keep aligned


mask expectations
--------------------

8mm and 2mm look expected for mask and gap::

     36 HamamatsuMaskManager::HamamatsuMaskManager(const std::string& name)
     37     :
     38 #ifdef PMTSIM_STANDALONE
     39     m_objName(name),
     40 #else
     41     ToolBase(name),
     42 #endif
     43      logicMaskTail(NULL), physiMaskTail(NULL)
     44     , LAB(NULL), Acrylic(NULL), Water(NULL), AcrylicMask(NULL), Steel(NULL)
     45 {
     46 #ifdef PMTSIM_STANDALONE
     47     m_buffer_material = "Water" ;
     48     htop_thickness=8.*mm ;
     49     htop_gap=2.*mm ;
     50     requator_thickness=8.*mm ;
     51     requator_gap=2.*mm ;
     52     m_useRealSurface=true ;
     53     m_useRealMaskTail=true ;
     54     m_useMaskTailOpSurface=true ;
     55 #else
     56     declProp("BufferMaterial", m_buffer_material = "Water");
     57     declProp("TopThickness", htop_thickness=8.*mm);
     58     declProp("TopGap", htop_gap=2.*mm); // gap between PMT and Mask at top
     59 


expectations
-------------

How many points would be expected for masked pyrex PMT in water, 5::


      TO             BT       BT BT    SD
                  wa |ac      |wa| py  | va
                     |        |  |     |   
      +--->          | 8mm    |2 |  5mm|   
                     |        |  |  ?  |   
                     |        |  |     |   
                     |        |  |     |   
                     ^           ^
                                 B:double skin at PMT
                                 (recall a technical 0.001)
             

TODO: look into mask 


issue 1 : symptoms 
------------------------

a step dist::

    In [49]: rdist_(a,0)[:10]
    Out[49]: array([790.062, 790.702, 790.425, 790.766, 791.122, 600.775, 792.16 , 790.604, 790.293, 790.777], dtype=float32)

    In [50]: rdist_(a,1)[:10]
    Out[50]: array([    8.   ,     8.011,     8.006,     8.012,     8.019, 19044.775,     8.036,     8.011,     8.006,     8.013], dtype=float32)

    In [51]: rdist_(a,2)[:10]
    Out[51]: array([1.999, 2.004, 2.003, 2.003, 2.005, 0.   , 2.011, 2.002, 2.   , 2.004], dtype=float32)

    In [52]: rdist_(a,3)[:10]
    Out[52]: array([5.054, 5.6  , 5.361, 5.652, 5.957, 0.   , 6.853, 5.516, 5.251, 5.664], dtype=float32)


TODO: axial photon to check the pyrex dist without angles complications

b step dist (one clear degenerate)::

    In [42]: rdist_(b,0)[:10]   ## distance between point 0 and 1 
    Out[42]: array([789.95 , 789.949, 789.95 , 789.951, 789.95 , 600.776, 789.951, 789.95 , 789.95 , 789.95 ], dtype=float32)

    In [43]: rdist_(b,1)[:10]   
    Out[43]: array([    0.112,     0.753,     0.475,     0.815,     1.172, 19044.777,     2.209,     0.655,     0.343,     0.827], dtype=float32)

    In [44]: rdist_(b,2)[:10]
    Out[44]: array([8.001, 8.013, 8.007, 8.013, 8.019, 0.   , 8.036, 8.01 , 8.005, 8.013], dtype=float32)

    In [45]: rdist_(b,3)[:10]
    Out[45]: array([1.999, 2.003, 2.001, 2.003, 2.005, 0.   , 2.011, 2.002, 2.001, 2.003], dtype=float32)

    In [46]: rdist_(b,4)[:10]
    Out[46]: array([0.001, 0.001, 0.001, 0.002, 0.001, 0.   , 0.001, 0.001, 0.001, 0.002], dtype=float32)
    ## technical PMT skin surface : this is the one that Opticks geometry skips 

    In [47]: rdist_(b,5)[:10]
    Out[47]: array([5.   , 5.008, 5.004, 5.008, 5.012, 0.   , 5.024, 5.007, 5.004, 5.008], dtype=float32)





point positions

A 5 points::

    In [36]: a.record[0,:5,0]
    Out[36]: 
    array([[-11465.841,   9041.017,  11268.351,      0.   ],
           [-11956.769,   9428.393,  11751.184,      3.651],
           [-11961.729,   9432.32 ,  11756.08 ,      3.691],
           [-11962.971,   9433.301,  11757.302,      3.701],
           [-11966.104,   9435.781,  11760.395,      3.726]], dtype=float32)

B 7 points::

    In [37]: b.record[0,:7,0]
    Out[37]: 
    array([[-11465.841,   9041.017,  11268.351,      0.   ],
           [-11956.698,   9428.338,  11751.115,      3.653],
           [-11956.768,   9428.393,  11751.184,      3.653],
           [-11961.728,   9432.32 ,  11756.081,      3.694],
           [-11962.97 ,   9433.301,  11757.303,      3.703],
           [-11962.971,   9433.302,  11757.303,      3.703],
           [-11966.071,   9435.756,  11760.363,      3.728]], dtype=float32)



A gives the expected 5 points, with 3 BT::

    In [6]: seqhis_(a.seq[:10,0])
    Out[6]: 
    ['TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO AB',
     'TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO BT BT BT SD',
     'TO BT BT BT SD']

B giving 7 points, 5 BT::

    In [7]: seqhis_(b.seq[:10,0])
    Out[7]: 
    ['TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO AB',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD',
     'TO BT BT BT BT BT SD']


  


local frame points
----------------------


In local frame, plot em::

    In [1]: a_lpos
    Out[1]: 
    array([[ -4.295,   4.959, 989.999,   1.   ],
           [ -4.295,   4.959, 199.937,   1.   ],
           [ -4.286,   4.948, 191.936,   1.   ],
           [ -4.286,   4.948, 189.937,   1.   ],
           [ -4.28 ,   4.942, 184.884,   1.   ]])

    In [2]: b_lpos
    Out[2]: 
    array([[ -4.295,   4.959, 989.999,   1.   ],
           [ -4.295,   4.959, 200.049,   1.   ],
           [ -4.295,   4.959, 199.937,   1.   ],
           [ -4.285,   4.948, 191.936,   1.   ],
           [ -4.285,   4.947, 189.937,   1.   ],
           [ -4.286,   4.947, 189.936,   1.   ],
           [ -4.28 ,   4.941, 184.936,   1.   ]])

    In [3]: a.inphoton[0]
    Out[3]: 
    array([[-11465.841,   9041.017,  11268.351,      0.   ],
           [    -0.621,      0.49 ,      0.611,      0.   ],
           [     0.043,      0.8  ,     -0.598,    501.   ],
           [     0.   ,      0.   ,      0.   ,      0.   ]], dtype=float32)



TODO: store the untransformed input photon also for debugging 


cfg4 (old worlkflow) microStep
----------------------------------

::

    epsilon:cfg4 blyth$ grep microStep *.*
    CRecorder.cc:    m_microStep_mm(0.004),              //  see notes/issues/ok_lacks_SI-4BT-SD.rst
    CRecorder.cc:    m_suppress_same_material_microStep(true), 
    CRecorder.cc:    m_suppress_all_microStep(true), 
    CRecorder.cc:        bool microStep = step_mm <= m_microStep_mm ; 
    CRecorder.cc:        bool suppress_microStep = false ; 
    CRecorder.cc:        if(m_suppress_same_material_microStep ) suppress_microStep = premat == postmat && microStep ;
    CRecorder.cc:        if(m_suppress_all_microStep )           suppress_microStep = microStep ;       
    CRecorder.cc:        // suppress_all_microStep trumps suppress_same_material_microStep
    CRecorder.cc:        //if(postFlag == 0 || suppress_microStep )
    CRecorder.cc:                << " suppress_microStep " << ( suppress_microStep ? "YES" : "n" )
    CRecorder.cc:                << " m_microStep_mm " << m_microStep_mm 
    CRecorder.cc:        bool postSkip = ( boundary_status == Ds::StepTooSmall || suppress_microStep ) && !lastPost  ;  
    CRecorder.cc:        bool postSkip = ( boundary_status == StepTooSmall || suppress_microStep ) && !lastPost  ;  
    CRecorder.cc:        if(postSkip)      m_state._step_action |= CAction::POST_SKIP ;    // StepTooSmall or microStep being suppressed
    CRecorder.hh:        double             m_microStep_mm ; 
    CRecorder.hh:        bool               m_suppress_same_material_microStep ; 
    CRecorder.hh:        bool               m_suppress_all_microStep ; 
    epsilon:cfg4 blyth$ 
    epsilon:cfg4 blyth$ 



::

    094 CRecorder::CRecorder(CCtx& ctx)
     95     :
     96     m_ctx(ctx),
     97     m_ok(m_ctx.getOpticks()),
     98     m_microStep_mm(0.004),              //  see notes/issues/ok_lacks_SI-4BT-SD.rst
     99     m_suppress_same_material_microStep(true),
    100     m_suppress_all_microStep(true),


    479 void CRecorder::postTrackWriteSteps()
    ...
    528         G4ThreeVector delta = step->GetDeltaPosition();
    529         double        step_mm = delta.mag()/mm  ;
    530         bool microStep = step_mm <= m_microStep_mm ;
    ...
    554         bool suppress_microStep = false ;
    555         if(m_suppress_same_material_microStep ) suppress_microStep = premat == postmat && microStep ;
    556         if(m_suppress_all_microStep )           suppress_microStep = microStep ;
    557         // suppress_all_microStep trumps suppress_same_material_microStep
    ...
    590 #ifdef USE_CUSTOM_BOUNDARY
    591         bool postSkip = ( boundary_status == Ds::StepTooSmall || suppress_microStep ) && !lastPost  ;
    592         bool matSwap = next_boundary_status == Ds::StepTooSmall ;
    593 #else
    594         bool postSkip = ( boundary_status == StepTooSmall || suppress_microStep ) && !lastPost  ;
    595         bool matSwap = next_boundary_status == StepTooSmall ;
    596 #endif
    ...
    670         else     // after 1st step just POST_SAVE 
    671         {
    672             if(!postSkip && !done)
    673             {
    674                 m_state._step_action |= CAction::POST_SAVE ;
    675 
    676                 done = WriteStepPoint( post, postFlag, u_postmat, boundary_status, POST, false );
    677 
    678                 if(done) m_state._step_action |= CAction::POST_DONE ;
    679             }
    680         }


TODO : microStep in U4Recorder with random stream rewind
--------------------------------------------------------------

* how to rewind, need to save the flat cursor at starts so can rewind back to that position in stream 
  when a point gets skipped  


