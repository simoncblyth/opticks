review_alignment
====================

State of play
---------------

* using input photons more advanced using "--align" option
* using input gensteps incomplete, only CerenkovGenerator
  and there are material consistency issues :doc:`ckm-okg4-material-rindex-mismatch`




G4Navigator
------------

::

   g4-cc ZeroStep

    
g4-cls G4Navigator::

    290   inline G4int SeverityOfZeroStepping( G4int* noZeroSteps ) const;
    291     // Report on severity of error and number of zero steps,
    292     // in case Navigator is stuck and is returning zero steps.
    293     // Values: 1 (small problem),  5 (correcting), 
    294     //         9 (ready to abandon), 10 (abandoned)
    ...
    437   // Count zero steps - as one or two can occur due to changing momentum at
    438   //                    a boundary or at an edge common between volumes
    439   //                  - several are likely a problem in the geometry
    440   //                    description or in the navigation
    441   //
    442   G4bool fLastStepWasZero;
    443     // Whether the last ComputeStep moved Zero. Used to check for edges.
    444 
    445   G4bool fLocatedOnEdge;
    446     // Whether the Navigator has detected an edge
    447   G4int fNumberZeroSteps;
    448     // Number of preceding moves that were Zero. Reset to 0 after finite step
    449   G4int fActionThreshold_NoZeroSteps;
    450     // After this many failed/zero steps, act (push etc) 
    451   G4int fAbandonThreshold_NoZeroSteps;
    452     // After this many failed/zero steps, abandon track



    437   // Count zero steps - as one or two can occur due to changing momentum at
    438   //                    a boundary or at an edge common between volumes
    439   //                  - several are likely a problem in the geometry
    440   //                    description or in the navigation
    441   //
    442   G4bool fLastStepWasZero;
    443     // Whether the last ComputeStep moved Zero. Used to check for edges.
    444 
    445   G4bool fLocatedOnEdge;
    446     // Whether the Navigator has detected an edge
    447   G4int fNumberZeroSteps;
    448     // Number of preceding moves that were Zero. Reset to 0 after finite step
    449   G4int fActionThreshold_NoZeroSteps;
    450     // After this many failed/zero steps, act (push etc) 
    451   G4int fAbandonThreshold_NoZeroSteps;
    452     // After this many failed/zero steps, abandon track
    453 
    454   G4ThreeVector  fPreviousSftOrigin;
    455   G4double       fPreviousSafety;
    456     // Memory of last safety origin & value. Used in ComputeStep to ensure
    457     // that origin of current Step is in the same volume as the point of the
    458     // last relocation


    0058 G4Navigator::G4Navigator()
      59   : fWasLimitedByGeometry(false), fVerbose(0),
      60     fTopPhysical(0), fCheck(false), fPushed(false), fWarnPush(true)
      61 {
      62   fActive= false;
      63   fLastTriedStepComputation= false;
      64 
      65   ResetStackAndState();
      66     // Initialises also all 
      67     // - exit / entry flags
      68     // - flags & variables for exit normals
      69     // - zero step counters
      70     // - blocked volume 
      71 
      72   fActionThreshold_NoZeroSteps  = 10;
      73   fAbandonThreshold_NoZeroSteps = 25;
      74 



    444 // ********************************************************************
    445 // SeverityOfZeroStepping
    446 //
    447 // Reports on severity of error in case Navigator is stuck
    448 // and is returning zero steps
    449 // ********************************************************************
    450 //
    451 inline
    452 G4int G4Navigator::SeverityOfZeroStepping( G4int* noZeroSteps ) const
    453 { 
    454   G4int severity=0, noZeros= fNumberZeroSteps;
    455   if( noZeroSteps) *noZeroSteps = fNumberZeroSteps;
    456   
    457   if( noZeros >= fAbandonThreshold_NoZeroSteps )
    458   { 
    459     severity = 10;
    460   }
    461   if( noZeros > 0 && noZeros < fActionThreshold_NoZeroSteps )
    462   { 
    463     severity =  5 * noZeros / fActionThreshold_NoZeroSteps;
    464   }
    465   else if( noZeros == fActionThreshold_NoZeroSteps )
    466   { 
    467     severity =  5;
    468   }
    469   else if( noZeros >= fAbandonThreshold_NoZeroSteps - 2 )
    470   { 
    471     severity =  9;
    472   }
    473   else if( noZeros < fAbandonThreshold_NoZeroSteps - 2 )
    474   { 
    475     severity =  5 + 4 * (noZeros-fAbandonThreshold_NoZeroSteps)
    476                       / fActionThreshold_NoZeroSteps;
    477   }
    478   return severity;



* TODO: lldb look at ZeroSteps

::

    0972   // Count zero steps - one can occur due to changing momentum at a boundary
     973   //                  - one, two (or a few) can occur at common edges between
     974   //                    volumes
     975   //                  - more than two is likely a problem in the geometry
     976   //                    description or the Navigation 
     977 
     978   // Rule of thumb: likely at an Edge if two consecutive steps are zero,
     979   //                because at least two candidate volumes must have been
     980   //                checked
     981   //
     982   fLocatedOnEdge   = fLastStepWasZero && (Step==0.0);
     983   fLastStepWasZero = (Step<fMinStep);
     984   if (fPushed)  { fPushed = fLastStepWasZero; }
     985 
     986   // Handle large number of consecutive zero steps
     987   //
     988   if ( fLastStepWasZero )
     989   {
     990     fNumberZeroSteps++;
     991 #ifdef G4DEBUG_NAVIGATION
     992     if( fNumberZeroSteps > 1 )
     993     {
     994        G4cout << "G4Navigator::ComputeStep(): another 'zero' step, # "
     995               << fNumberZeroSteps
     996               << ", at " << pGlobalpoint
     997               << ", in volume " << motherPhysical->GetName()
     998               << ", nav-comp-step calls # " << sNavCScalls
     999               << ", Step= " << Step
    1000               << G4endl;
    1001     }
    1002 #endif

    1003     if( fNumberZeroSteps > fActionThreshold_NoZeroSteps-1 )
    1004     {  
    1005        // Act to recover this stuck track. Pushing it along direction
    1006        //
    1007        Step += 100*kCarTolerance;
    1008 #ifdef G4VERBOSE
    1009        if ((!fPushed) && (fWarnPush))
    1010        { 
    1011          std::ostringstream message;
    1012          message.precision(16);
    1013          message << "Track stuck or not moving." << G4endl
    1014                  << "          Track stuck, not moving for "
    1015                  << fNumberZeroSteps << " steps" << G4endl
    1016                  << "          in volume -" << motherPhysical->GetName()
    1017                  << "- at point " << pGlobalpoint
    1018                  << " (local point " << newLocalPoint << ")" << G4endl
    1019                  << "          direction: " << pDirection
    1020                  << " (local direction: " << localDirection << ")." << G4endl
    1021                  << "          Potential geometry or navigation problem !"
    1022                  << G4endl
    1023                  << "          Trying pushing it of " << Step << " mm ...";
    1024          G4Exception("G4Navigator::ComputeStep()", "GeomNav1002",
    1025                      JustWarning, message, "Potential overlap in geometry!");
    1026        }
    1027 #endif 
    1028        fPushed = true;
    1029     }








OKG4Mgr alignment with input photons
---------------------------------------------

::

    102 /**
    103 OKG4Mgr::propagate_
    104 ---------------------
    105 
    106 Hmm propagate implies just photons to me, so this name 
    107 is misleading as it does a G4 beamOn with the hooked up 
    108 CSource subclass providing the primaries, which can be 
    109 photons but not necessarily. 
    110 
    111 Normally the G4 propagation is done first, because 
    112 gensteps eg from G4Gun can then be passed to Opticks.
    113 However with RNG-aligned testing using "--align" option
    114 which uses emitconfig CPU generated photons there is 
    115 no need to do G4 first. Actually it is more convenient
    116 for Opticks to go first in order to allow access to the ucf.py 
    117 parsed  kernel pindex log during lldb python scripted G4 debugging.
    118  
    119 Hmm it would be cleaner if m_gen was in charge if the branching 
    120 here as its kinda similar to initSourceCode.
    121 
    122 
    123 Notice the different genstep handling between this and OKMgr 
    124 because this has G4 available, so gensteps can come from the
    125 horses mouth.
    126 
    127 
    128 
    129 **/



ckm 
-----

* :doc:`OKG4Test_direct_two_executable_shakedown`

* :doc:`strategy_for_Cerenkov_Scintillation_alignment`

  * Apply CAlignEngine to CerenkovMinimal+G4Opticks 


* https://bitbucket.org/simoncblyth/opticks/commits/all?page=8


To find relevant commits::

    hg flog notes/issues/strategy_for_Cerenkov_Scintillation_alignment.rst

    hg flog cfg4/CAlignEngine.cc | grep rst


Stage have got to... 

* not wanting to instrument CerenkovMinimal with CFG4 complexity 
* but need the complexity for step by step comparison 



NEXT STEP : GET CFG4 TO RUN FROM GENSTEPS USING CCerenkovGenerator 


::

    blyth@localhost issues]$ l -1 *ckm*
    -rw-rw-r--. 1 blyth blyth 38363 May 29 16:59 ckm_cerenkov_generation_align_small_quantized_deviation_g4_g4.rst
    -rw-rw-r--. 1 blyth blyth 19015 May 28 21:53 ckm-analysis-shakedown.rst
    -rw-rw-r--. 1 blyth blyth  3903 May 27 23:09 ckm-viz-noshow-photon-first-steps.rst
    -rw-rw-r--. 1 blyth blyth  7445 Apr  1 18:38 ckm_revival_improve_error_handling_of_missing_rng_seq.rst
    -rw-rw-r--. 1 blyth blyth 15886 Mar 20 17:29 ckm-okg4-natural-fail.rst
    -rw-rw-r--. 1 blyth blyth  4960 Oct 15  2018 ckm_cerenkov_generation_align_g4_ok_deviations.rst
    -rw-rw-r--. 1 blyth blyth 24171 Oct 15  2018 ckm_cerenkov_generation_align.rst




