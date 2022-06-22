ideas_on_random_alignment_in_new_workflow
===========================================

* from :doc:`U4RecorderTest_cf_CXRaindropTest`

Aligning the simulations requires:

1. same random streams 
2. same randoms get used for the same purposes in the two simulations. 

How to do that.

1. devise simtags enumeration for random consumptions that can be derived eg from the backtrace (or from GPU logging) 
   collect these names into simstream arrays 

   * whilst developing could also collect the random values to check are getting them all 
   * advantage of doing this from backtraces is that it can be automated, so can do for millions of photons

2. some simple code to read the two simstreams and present them together, 
   so can see where the "zippers" are not aligned  

3. study G4 and Opticks code to find "burn" random consumptions that are actually not being used, insert corresponding 
   burns (with instrumentation enumeration tags) and possibly reorder curand calls into GPU code 
   to get the consumption to line up 




WIP : Getting CXRaindropTest to run again with tag and flat collection
-------------------------------------------------------------------------



U4RecorderTest
-----------------

8 randoms from "TO BT SA"::

    2022-06-22 14:26:06.240 INFO  [28940848] [SEvt::beginPhoton@479] spho (gs:ix:id:gn   0   0    0  0)
    2022-06-22 14:26:06.241 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022 stack  2 ScintDiscreteReset
    2022-06-22 14:26:06.242 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845 stack  6 BoundaryDiscreteReset
    2022-06-22 14:26:06.243 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    2 idx    2 d    0.51701 stack  4 RayleighDiscreteReset
    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    3 idx    3 d    0.15699 stack  3 AbsorptionDiscreteReset
    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    4 idx    4 d    0.07137 stack  8 BoundaryBurn
     DiDi0  pidx      0 rand    0.07137 theReflectivity     1.0000 rand > theReflectivity  0
    DiDi.pidx    0 PIDX   -1 OldMomentum (   -0.77425   -0.24520    0.58345) OldPolarization (   -0.60182    0.00000   -0.79863) cost1    1.00000 Rindex1    1.35297 Rindex2    1.00027 sint1    0.00000 sint2    0.00000
    //DiDi pidx      0 : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  
    //DiDi pidx      0 : TransCoeff     0.9775 
    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    5 idx    5 d    0.46251 stack  7 BoundaryDiDi
    //InstrumentedG4OpBoundaryProcess::G4BooleanRand pidx      0 prob    0.97754 u    0.46251 u < prob 1 
    //DiDi pidx      0 : TRANSMIT 
    //DiDi pidx    0 : TRANSMIT NewMom (   -0.7742    -0.2452     0.5835) NewPol (   -0.6018     0.0000    -0.7986) 
    2022-06-22 14:26:06.244 INFO  [28940848] [SEvt::pointPhoton@730]  idx 0 bounce 0 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 14:26:06.244 INFO  [28940848] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis                d nib  1 TO
    2022-06-22 14:26:06.244 INFO  [28940848] [SEvt::pointPhoton@730]  idx 0 bounce 1 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 14:26:06.244 INFO  [28940848] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis               cd nib  2 TO BT
    2022-06-22 14:26:06.245 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    6 idx    6 d    0.22764 stack  2 ScintDiscreteReset
    2022-06-22 14:26:06.246 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    7 idx    7 d    0.32936 stack  6 BoundaryDiscreteReset
    2022-06-22 14:26:06.246 INFO  [28940848] [SEvt::pointPhoton@730]  idx 0 bounce 2 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 14:26:06.246 INFO  [28940848] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis              8cd nib  3 TO BT SA


What are the 8 doing for "TO BT SA":

    2022-06-22 14:26:06.241 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022 stack  2 ScintDiscreteReset
    2022-06-22 14:26:06.242 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845 stack  6 BoundaryDiscreteReset
    2022-06-22 14:26:06.243 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    2 idx    2 d    0.51701 stack  4 RayleighDiscreteReset
    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    3 idx    3 d    0.15699 stack  3 AbsorptionDiscreteReset
    # decide which process wins step 0->1 : its boundary 

    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    4 idx    4 d    0.07137 stack  8 BoundaryBurn
    2022-06-22 14:26:06.244 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    5 idx    5 d    0.46251 stack  7 BoundaryDiDi
    # proceed with the boundary process : 2 randoms to decide between BT/BR 

    2022-06-22 14:26:06.245 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    6 idx    6 d    0.22764 stack  2 ScintDiscreteReset
    2022-06-22 14:26:06.246 INFO  [28940848] [U4Random::flat@425]  m_seq_index    0 m_seq_nv  256 cursor    7 idx    7 d    0.32936 stack  6 BoundaryDiscreteReset
    # deciding on winner of step 1->2 : ends by getting killed by NoRindex on the Rock 


Hmm, how does Opticks side use 16, for this ?



Distinguising the processes from the backtraces ?
--------------------------------------------------

RestDiscreteReset
    must be scintillation as thats the only RestDiscrete process around

DiscreteReset
    one of three : G4OpAbsoption G4OpRayleigh InstrumentedG4OpBoundaryProcess

    U4Random::flat
    G4VProcess::ResetNumberOfInteractionLengthLeft
    G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4VProcess::PostStepGPIL
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping



Can I shim G4VProcess::ResetNumberOfInteractionLengthLeft to get the process name to appear in the backtrace ?::

    class U4_API InstrumentedG4OpBoundaryProcess : public G4VDiscreteProcess

    class G4VDiscreteProcess : public G4VProcess


YES, adding shim works to make the backtrace easy to U4Stack::Classify::

    111 class DsG4Scintillation : public G4VRestDiscreteProcess, public G4UImessenger
    ...
    119 public:
    120 #ifdef DEBUG_TAG
    121      // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
    122      void ResetNumberOfInteractionLengthLeft(){  G4VProcess::ResetNumberOfInteractionLengthLeft() ; }
    123 #endif
    124 

    136 class U4_API InstrumentedG4OpBoundaryProcess : public G4VDiscreteProcess
    137 {
    ...
    144 public:
    145 #ifdef DEBUG_TAG
    146         // Shim makes process classname appear in SBacktrace.h enabling U4Random::flat/U4Stack::Classify
    147         void ResetNumberOfInteractionLengthLeft(){ G4VProcess::ResetNumberOfInteractionLengthLeft(); }
    148 #endif
    149 

DONE: addes Shims to distinguish G4OpAbsorption from G4OpRayleigh



g4-cls G4VProcess::

    303  public: // with description
    304       virtual void      ResetNumberOfInteractionLengthLeft();
    305      // reset (determine the value of)NumberOfInteractionLengthLeft
    306 
    307       G4double GetNumberOfInteractionLengthLeft() const;
    308      // get NumberOfInteractionLengthLeft
    309 
    310       G4double GetTotalNumberOfInteractionLengthTraversed() const;
    311      // get NumberOfInteractionLength 
    312      //   after  ResetNumberOfInteractionLengthLeft is invoked
    313 
    314  protected:  // with description
    315      void      SubtractNumberOfInteractionLengthLeft(
    316                   G4double previousStepSize
    317                                 );
    318      // subtract NumberOfInteractionLengthLeft by the value corresponding to 
    319      // previousStepSize      
    320 
    321      void      ClearNumberOfInteractionLengthLeft();
    322      // clear NumberOfInteractionLengthLeft 
    323      // !!! This method should be at the end of PostStepDoIt()
    324      // !!! and AtRestDoIt
    325 

    096 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     97 {
     98   theNumberOfInteractionLengthLeft =  -1.*G4Log( G4UniformRand() );
     99   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
    100 }




::

    2022-06-22 11:20:34.253 INFO  [28802444] [SEvt::beginPhoton@479] spho (gs:ix:id:gn   0   0    0  0)
    2022-06-22 11:20:34.254 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022 stack  1 RestDiscreteReset
    2022-06-22 11:20:34.255 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845 stack  2 DiscreteReset
    2022-06-22 11:20:34.255 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    2 idx    2 d    0.51701 stack  2 DiscreteReset
    2022-06-22 11:20:34.256 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    3 idx    3 d    0.15699 stack  2 DiscreteReset
    2022-06-22 11:20:34.256 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    4 idx    4 d    0.07137 stack  4 BoundaryBurn
     DiDi0  pidx      0 rand    0.07137 theReflectivity     1.0000 rand > theReflectivity  0
    DiDi.pidx    0 PIDX   -1 OldMomentum (   -0.77425   -0.24520    0.58345) OldPolarization (   -0.60182    0.00000   -0.79863) cost1    1.00000 Rindex1    1.35297 Rindex2    1.00027 sint1    0.00000 sint2    0.00000
    //DiDi pidx      0 : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  
    //DiDi pidx      0 : TransCoeff     0.9775 
    2022-06-22 11:20:34.256 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    5 idx    5 d    0.46251 stack  3 BoundaryDiDi
    //InstrumentedG4OpBoundaryProcess::G4BooleanRand pidx      0 prob    0.97754 u    0.46251 u < prob 1 
    //DiDi pidx      0 : TRANSMIT 
    //DiDi pidx    0 : TRANSMIT NewMom (   -0.7742    -0.2452     0.5835) NewPol (   -0.6018     0.0000    -0.7986) 
    2022-06-22 11:20:34.257 INFO  [28802444] [SEvt::pointPhoton@730]  idx 0 bounce 0 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 11:20:34.257 INFO  [28802444] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis                d nib  1 TO
    2022-06-22 11:20:34.257 INFO  [28802444] [SEvt::pointPhoton@730]  idx 0 bounce 1 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 11:20:34.257 INFO  [28802444] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis               cd nib  2 TO BT
    2022-06-22 11:20:34.257 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    6 idx    6 d    0.22764 stack  1 RestDiscreteReset
    2022-06-22 11:20:34.258 INFO  [28802444] [U4Random::flat@424]  m_seq_index    0 m_seq_nv  256 cursor    7 idx    7 d    0.32936 stack  2 DiscreteReset
    2022-06-22 11:20:34.258 INFO  [28802444] [SEvt::pointPhoton@730]  idx 0 bounce 2 evt.max_record 10 evt.max_rec    10 evt.max_seq    10 evt.max_prd    10 evt.max_tag    24 evt.max_flat    24
    2022-06-22 11:20:34.258 INFO  [28802444] [SEvt::pointPhoton@751] spho (gs:ix:id:gn   0   0    0  0)  seqhis              8cd nib  3 TO BT SA
    2022-06-22 11:20:34.258 INFO  [28802444] [U4Random::setSequenceIndex@282]  index -1
    2022-06-22 11:20:34.258 INFO  [28802444] [SEvt::finalPhoton@776] spho (gs:ix:id:gn   0   0    0  0)
    2022-06-22 11:20:34.258 INFO  [28802444] [U4Recorder::EndOfEventAction@51] 





SBacktrace.h U4Stack.h classifying U4Random::flat backtraces to follow every random consumption
---------------------------------------------------------------------------------------------------

* TODO: LOOK INTO THE TAIL BURNS, ARE THEY ACTUALLY DOING ANYTHING ?
* TODO: investigate Geant4 process ordering to allow the stack enumeration to be translated into the stag.h enumeration  

* DONE: collect the stack tags and flat in G4 side using SEvt machinery 
  (even prior to enumeration translation), so can script the array alignment comparison



::

    2022-06-21 16:31:54.832 INFO  [28350265] [U4RecorderTest::GeneratePrimaries@111] [ mode I
    SGenerate::GeneratePhotons ph  <f8(10, 4, 4, )
    2022-06-21 16:31:54.832 INFO  [28350265] [U4RecorderTest::GeneratePrimaries@119] ]
    2022-06-21 16:31:54.832 INFO  [28350265] [U4Recorder::BeginOfEventAction@50] 
    2022-06-21 16:31:54.832 INFO  [28350265] [U4Random::setSequenceIndex@282]  index 9
    2022-06-21 16:31:54.835 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    0 idx 2304 d    0.51319 stack RestDiscreteReset
    2022-06-21 16:31:54.836 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    1 idx 2305 d    0.04284 stack DiscreteReset
    2022-06-21 16:31:54.837 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    2 idx 2306 d    0.95184 stack DiscreteReset
    2022-06-21 16:31:54.838 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    3 idx 2307 d    0.92588 stack DiscreteReset
    2022-06-21 16:31:54.838 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    4 idx 2308 d    0.25979 stack BoundaryBurn
     DiDi0  pidx      9 rand    0.25979 theReflectivity     1.0000 rand > theReflectivity  0
    DiDi.pidx    9 PIDX   -1 OldMomentum (   -0.50013    0.44970    0.74002) OldPolarization (   -0.82853    0.00000   -0.55994) cost1    1.00000 Rindex1    1.35297 Rindex2    1.00027 sint1    0.00000 sint2    0.00000
    //DiDi pidx      9 : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  
    //DiDi pidx      9 : TransCoeff     0.9775 
    2022-06-21 16:31:54.838 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    5 idx 2309 d    0.91341 stack BoundaryDiDi
    //InstrumentedG4OpBoundaryProcess::G4BooleanRand pidx      9 prob    0.97754 u    0.91341 u < prob 1 
    //DiDi pidx      9 : TRANSMIT 
    //DiDi pidx    9 : TRANSMIT NewMom (   -0.5001     0.4497     0.7400) NewPol (   -0.8285     0.0000    -0.5599) 
    2022-06-21 16:31:54.839 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    6 idx 2310 d    0.39325 stack RestDiscreteReset
    2022-06-21 16:31:54.840 INFO  [28350265] [U4Random::flat@423]  m_seq_index    9 m_seq_nv  256 cursor    7 idx 2311 d    0.83318 stack DiscreteReset
    2022-06-21 16:31:54.840 INFO  [28350265] [U4Random::setSequenceIndex@282]  index -1
    2022-06-21 16:31:54.840 INFO  [28350265] [U4Random::setSequenceIndex@282]  index 8
    2022-06-21 16:31:54.841 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    0 idx 2048 d    0.47022 stack RestDiscreteReset
    2022-06-21 16:31:54.842 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    1 idx 2049 d    0.48217 stack DiscreteReset
    2022-06-21 16:31:54.843 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    2 idx 2050 d    0.42791 stack DiscreteReset
    2022-06-21 16:31:54.844 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    3 idx 2051 d    0.44174 stack DiscreteReset
    2022-06-21 16:31:54.844 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    4 idx 2052 d    0.78041 stack BoundaryBurn
     DiDi0  pidx      8 rand    0.78041 theReflectivity     1.0000 rand > theReflectivity  0
    DiDi.pidx    8 PIDX   -1 OldMomentum (    0.80941   -0.18808    0.55631) OldPolarization (   -0.56642    0.00000    0.82412) cost1    1.00000 Rindex1    1.35297 Rindex2    1.00027 sint1    0.00000 sint2    0.00000
    //DiDi pidx      8 : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  
    //DiDi pidx      8 : TransCoeff     0.9775 
    2022-06-21 16:31:54.844 INFO  [28350265] [U4Random::flat@423]  m_seq_index    8 m_seq_nv  256 cursor    5 id



::

    2022-06-21 16:31:54.883 INFO  [28350265] [U4Random::setSequenceIndex@282]  index 0
    2022-06-21 16:31:54.884 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022 stack RestDiscreteReset
    2022-06-21 16:31:54.884 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845 stack DiscreteReset
    2022-06-21 16:31:54.885 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    2 idx    2 d    0.51701 stack DiscreteReset
    2022-06-21 16:31:54.886 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    3 idx    3 d    0.15699 stack DiscreteReset
    2022-06-21 16:31:54.886 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    4 idx    4 d    0.07137 stack BoundaryBurn
     DiDi0  pidx      0 rand    0.07137 theReflectivity     1.0000 rand > theReflectivity  0
    DiDi.pidx    0 PIDX   -1 OldMomentum (   -0.77425   -0.24520    0.58345) OldPolarization (   -0.60182    0.00000   -0.79863) cost1    1.00000 Rindex1    1.35297 Rindex2    1.00027 sint1    0.00000 sint2    0.00000
    //DiDi pidx      0 : NOT:sint1 > 0 : JACKSON NORMAL INCIDENCE  
    //DiDi pidx      0 : TransCoeff     0.9775 
    2022-06-21 16:31:54.886 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    5 idx    5 d    0.46251 stack BoundaryDiDi
    //InstrumentedG4OpBoundaryProcess::G4BooleanRand pidx      0 prob    0.97754 u    0.46251 u < prob 1 
    //DiDi pidx      0 : TRANSMIT 
    //DiDi pidx    0 : TRANSMIT NewMom (   -0.7742    -0.2452     0.5835) NewPol (   -0.6018     0.0000    -0.7986) 
    2022-06-21 16:31:54.887 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    6 idx    6 d    0.22764 stack RestDiscreteReset
    2022-06-21 16:31:54.888 INFO  [28350265] [U4Random::flat@423]  m_seq_index    0 m_seq_nv  256 cursor    7 idx    7 d    0.32936 stack DiscreteReset
    2022-06-21 16:31:54.888 INFO  [28350265] [U4Random::setSequenceIndex@282]  index -1
    2022-06-21 16:31:54.888 INFO  [28350265] [U4Recorder::EndOfEventAction@51] 


HMM: the qsim.h is consuming 16 (but g4 only 8) (this is probably why I previously used some extra reset to make the consumption more regular for each step point)::

    In [3]: t.flat[:,:17]                                                                                                                                                       
    Out[3]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.975, 0.547, 0.653, 0.23 , 0.   ],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.509, 0.089, 0.007, 0.954, 0.547, 0.825, 0.527, 0.93 , 0.   ],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.082, 0.807, 0.695, 0.618, 0.256, 0.214, 0.342, 0.224, 0.   ],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.491, 0.328, 0.911, 0.191, 0.964, 0.898, 0.624, 0.71 , 0.   ],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.079, 0.148, 0.599, 0.426, 0.243, 0.489, 0.41 , 0.668, 0.   ],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.361, 0.62 , 0.45 , 0.306, 0.503, 0.456, 0.552, 0.848, 0.   ],
           [0.667, 0.397, 0.158, 0.542, 0.706, 0.126, 0.154, 0.653, 0.38 , 0.855, 0.208, 0.09 , 0.701, 0.434, 0.106, 0.082, 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   ]], dtype=float32)

U4RecorderTest.sh G4 consuming only 8::

    In [4]: t.flat[:,:10]
    Out[4]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.   , 0.   ],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.   , 0.   ],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.   , 0.   ],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.   , 0.   ],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.   , 0.   ],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.   , 0.   ],
           [0.667, 0.397, 0.158, 0.542, 0.706, 0.126, 0.154, 0.653, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.   , 0.   ],
           [0.47 , 0.482, 0.428, 0.442, 0.78 , 0.859, 0.614, 0.802, 0.   , 0.   ],
           [0.513, 0.043, 0.952, 0.926, 0.26 , 0.913, 0.393, 0.833, 0.   , 0.   ]], dtype=float32)

    In [3]: st[:,:10]   ## these are currently the U4Stack::Classify enumeration (not the stag.h ones)
    Out[3]: 
    array([[1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0],
           [1, 2, 2, 2, 4, 3, 1, 2, 0, 0]], dtype=uint8)

::

     07 class stag(object):
      8     """
      9     # the below NSEQ, BITS, ... param need to correspond to stag.h static constexpr 
     10     """
     11     lptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
     12     PATH = "$OPTICKS_PREFIX/include/sysrap/stag.h" 
     13     
     14     NSEQ = 2
     15     BITS = 5
     16     MASK = ( 0x1 << BITS ) - 1
     17     SLOTMAX = 64//BITS
     18     SLOTS = SLOTMAX*NSEQ
     19     
     20     @classmethod
     21     def Split(cls, tag):
     22         st = np.zeros( (len(tag), cls.SLOTS), dtype=np.uint8 )
     23         for i in range(cls.NSEQ): 
     24             for j in range(cls.SLOTMAX):
     25                 st[:,i*cls.SLOTMAX+j] = (tag[:,i] >> (cls.BITS*j)) & cls.MASK
     26             pass
     27         pass
     28         return st



FIXED : NOT getting expected flat with mock_propagate
--------------------------------------------------------

::

    In [1]: t.flat                                                                                                                                                             

    In [4]: t.flat[:,:18]                                                                                                                                                      
    Out[4]: 
    array([[0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   , 0.   ]], dtype=float32)


Compare with qudarap/tests/rng_sequence.sh ana::

    In [6]: a.shape                                                                                                                                                             
    Out[6]: (100000, 16, 16)

    In [7]: aa = a.reshape(-1,16*16)        

    In [9]: aa[:8,:18]                                                                                                                                                          
    Out[9]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.975, 0.547, 0.653, 0.23 , 0.339, 0.761],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.509, 0.089, 0.007, 0.954, 0.547, 0.825, 0.527, 0.93 , 0.163, 0.785],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.082, 0.807, 0.695, 0.618, 0.256, 0.214, 0.342, 0.224, 0.524, 0.921],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.491, 0.328, 0.911, 0.191, 0.964, 0.898, 0.624, 0.71 , 0.341, 0.067],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.079, 0.148, 0.599, 0.426, 0.243, 0.489, 0.41 , 0.668, 0.627, 0.277],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.361, 0.62 , 0.45 , 0.306, 0.503, 0.456, 0.552, 0.848, 0.368, 0.928],
           [0.667, 0.397, 0.158, 0.542, 0.706, 0.126, 0.154, 0.653, 0.38 , 0.855, 0.208, 0.09 , 0.701, 0.434, 0.106, 0.082, 0.22 , 0.294],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.282, 0.076]], dtype=float32)


Are getting idx 7 flat repeated 8 times ? Dumping shows are seeing all the flat, but are stomping::

    //stagr::add slot 0 tag  1 flat     0.7402 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.9210 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.0390 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.9690 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.9251 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.4464 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.6673 SLOTS 24 
    //stagr::add slot 0 tag  1 flat     0.1099 SLOTS 24 


After rearranging "stagr tagr" to be on same footing as "sseq seq", "sphoton p" etc.. rather than as qsim.h member avoid the stomping and get 
the expected flat collection::

    In [3]: t.flat[:,:17]                                                                                                                                                       
    Out[3]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.975, 0.547, 0.653, 0.23 , 0.   ],
           [0.921, 0.46 , 0.333, 0.373, 0.49 , 0.567, 0.08 , 0.233, 0.509, 0.089, 0.007, 0.954, 0.547, 0.825, 0.527, 0.93 , 0.   ],
           [0.039, 0.25 , 0.184, 0.962, 0.521, 0.94 , 0.831, 0.41 , 0.082, 0.807, 0.695, 0.618, 0.256, 0.214, 0.342, 0.224, 0.   ],
           [0.969, 0.495, 0.673, 0.563, 0.12 , 0.976, 0.136, 0.589, 0.491, 0.328, 0.911, 0.191, 0.964, 0.898, 0.624, 0.71 , 0.   ],
           [0.925, 0.053, 0.163, 0.89 , 0.567, 0.241, 0.494, 0.321, 0.079, 0.148, 0.599, 0.426, 0.243, 0.489, 0.41 , 0.668, 0.   ],
           [0.446, 0.338, 0.207, 0.985, 0.403, 0.178, 0.46 , 0.16 , 0.361, 0.62 , 0.45 , 0.306, 0.503, 0.456, 0.552, 0.848, 0.   ],
           [0.667, 0.397, 0.158, 0.542, 0.706, 0.126, 0.154, 0.653, 0.38 , 0.855, 0.208, 0.09 , 0.701, 0.434, 0.106, 0.082, 0.   ],
           [0.11 , 0.874, 0.981, 0.967, 0.162, 0.428, 0.931, 0.01 , 0.846, 0.38 , 0.812, 0.152, 0.273, 0.413, 0.786, 0.087, 0.   ]], dtype=float32)



WIP : sysrap/stag.h for tagging all random consumption
---------------------------------------------------------

::

    In [7]: t.tag[:,0] & 0x1f                                                                                                                                
    Out[7]: array([1, 1, 1, 1, 1, 1, 1, 1], dtype=uint64)      to_sc

    In [8]: ( t.tag[:,0] >> 5 ) & 0x1f                                                                                                                       
    Out[8]: array([2, 2, 2, 2, 2, 2, 2, 2], dtype=uint64)      to_ab

    In [9]: ( t.tag[:,0] >> 2*5 ) & 0x1f                                                                                                                     
    Out[9]: array([9, 9, 9, 9, 9, 9, 9, 9], dtype=uint64)      at_bo

    In [10]: ( t.tag[:,0] >> 3*5 ) & 0x1f                                                                                                                    
    Out[10]: array([10, 10, 10, 10, 10, 10, 10, 10], dtype=uint64)   at_rf



    In [11]: ( t.tag[:,0] >> 4*5 ) & 0x1f                                                                                                                    
    Out[11]: array([1, 1, 1, 1, 1, 1, 1, 1], dtype=uint64)       

    In [12]: ( t.tag[:,0] >> 5*5 ) & 0x1f                                                                                                                    
    Out[12]: array([2, 2, 2, 2, 2, 2, 2, 2], dtype=uint64)

    In [13]: ( t.tag[:,0] >> 6*5 ) & 0x1f                                                                                                                    
    Out[13]: array([9, 9, 9, 9, 9, 9, 9, 9], dtype=uint64)

    In [14]: ( t.tag[:,0] >> 7*5 ) & 0x1f                                                                                                                    
    Out[14]: array([10, 10, 10, 10, 10, 10, 10, 10], dtype=uint64)




    In [15]: ( t.tag[:,0] >> 8*5 ) & 0x1f                                                                                                                    
    Out[15]: array([1, 1, 1, 1, 1, 1, 1, 1], dtype=uint64)

    In [16]: ( t.tag[:,0] >> 9*5 ) & 0x1f                                                                                                                    
    Out[16]: array([2, 2, 2, 2, 2, 2, 2, 2], dtype=uint64)

    In [17]: ( t.tag[:,0] >> 10*5 ) & 0x1f                                                                                                                   
    Out[17]: array([9, 9, 9, 9, 9, 9, 9, 9], dtype=uint64)

    In [18]: ( t.tag[:,0] >> 11*5 ) & 0x1f                                                                                                                   
    Out[18]: array([10, 10, 10, 10, 10, 10, 10, 10], dtype=uint64)



    In [19]: ( t.tag[:,0] >> 12*5 ) & 0x1f                                                                                                                    
    Out[19]: array([0, 0, 0, 0, 0, 0, 0, 0], dtype=uint64)    ## HMM : AM I SKIPPING THE TOP SLOT ?

    In [20]: 12*5                                                                                                                                            
    Out[20]: 60

    In [21]: ( t.tag[:,1] >> 1*5 ) & 0x1f                                                                                                                    
    Out[21]: array([2, 2, 2, 2, 2, 2, 2, 2], dtype=uint64)

    In [22]: ( t.tag[:,1] >> 0*5 ) & 0x1f                                                                                                                    
    Out[22]: array([1, 1, 1, 1, 1, 1, 1, 1], dtype=uint64)

    In [23]: ( t.tag[:,1] >> 1*5 ) & 0x1f                                                                                                                    
    Out[23]: array([2, 2, 2, 2, 2, 2, 2, 2], dtype=uint64)

    In [24]: ( t.tag[:,1] >> 2*5 ) & 0x1f                                                                                                                    
    Out[24]: array([11, 11, 11, 11, 11, 11, 11, 11], dtype=uint64)

    In [25]: ( t.tag[:,1] >> 3*5 ) & 0x1f                                                                                                                    
    Out[25]: array([12, 12, 12, 12, 12, 12, 12, 12], dtype=uint64)

    In [26]: ( t.tag[:,1] >> 4*5 ) & 0x1f                                                                                                                    
    Out[26]: array([0, 0, 0, 0, 0, 0, 0, 0], dtype=uint64)




GPU side simstream
---------------------

* doing this from GPU logfile parsing is inherently limited to small stats

* would be good to run the GPU code on the CPU, so could use same SBacktrace machinery 

  * BUT that is a lot of work to setup, requiring prd and state captures or mocking texture lookups CPU side 

* GPU side are in control of all the code doing the consumption so can devise an enumeration for all 
  the curand_uniform callsite and write those enumerations into GPU side callsite/simstream array 

::

    epsilon:qudarap blyth$ grep curand_uniform qsim.h | wc -l 
          23

* if the number of active callsite were less than 16 it would be convenient for nibble packing 
* this enumeration should be reusable CPU side : it can have GPU side natural names eg::

     to_boundary_SI_burn 
     to_boundary_AB
     to_boundary_SC 

* hmm: can use same machinery that sseq does if less than 16 



CPU side simstream : many consumptions from G4 internals : so have to use SBacktrace for a complete picture
----------------------------------------------------------------------------------------------------------------

Review the start of the consumption deciding on the winning process for a step (~5 consumptions)

* :doc:`G4SteppingManager_DefinePhysicalStepLength`



On the CPU side SBacktrace.hh provides an automated way to collect backtraces, eg::

   U4Random_select=-1,0,-1,1 U4Random_select_action=backtrace ./U4RecorderTest.sh run
       ##  dump the backtrace for the first and second random consumption "cursor 0 and 1" of all photons pidx:"-1" 

::

    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@416]  m_seq_index    0 m_seq_nv  256 cursor    0 idx    0 d    0.74022
    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@430] U4Random_select -1,0,-1,1 m_select->size 4 (-1,0) YES  (-1,1) NO 
    SBacktrace::Dump addrlen 17
    SFrames..
    0   libSysRap.dylib                     0x0000000111bf7c7b SBacktrace::Dump(std::__1::basic_ostream<char, std::__1::char_traits<char> >&)                       + 107      
    1   libSysRap.dylib                     0x0000000111bf7bfb SBacktrace::Dump()                                                                                   + 27       
    2   libU4.dylib                         0x000000010c18b53c U4Random::flat()                                                                                     + 2348     
    3   libG4processes.dylib                0x000000010f6a96da G4VProcess::ResetNumberOfInteractionLengthLeft()                                                     + 42       
    4   libG4processes.dylib                0x000000010f6abd0b G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) + 91       
    5   libG4tracking.dylib                 0x000000010deffff0 G4VProcess::PostStepGPIL(G4Track const&, double, G4ForceCondition*)                                  + 80       
    6   libG4tracking.dylib                 0x000000010deffa1a G4SteppingManager::DefinePhysicalStepLength()                                                        + 298      
    7   libG4tracking.dylib                 0x000000010defcc3a G4SteppingManager::Stepping()                                                                        + 394      
    8   libG4tracking.dylib                 0x000000010df1386f G4TrackingManager::ProcessOneTrack(G4Track*)                                                         + 1679     
    9   libG4event.dylib                    0x000000010ddd871a G4EventManager::DoProcessing(G4Event*)                                                               + 3306     
    10  libG4event.dylib                    0x000000010ddd9c2f G4EventManager::ProcessOneEvent(G4Event*)                                                            + 47       
    11  libG4run.dylib                      0x000000010dce59e5 G4RunManager::ProcessOneEvent(int)                                                                   + 69       
    12  libG4run.dylib                      0x000000010dce5815 G4RunManager::DoEventLoop(int, char const*, int)                                                     + 101      
    13  libG4run.dylib                      0x000000010dce3cd1 G4RunManager::BeamOn(int, char const*, int)                                                          + 193      
    14  U4RecorderTest                      0x000000010c05a04a main + 1402
    15  libdyld.dylib                       0x00007fff72c44015 start + 1
    16  ???                                 0x0000000000000001 0x0 + 1
    2022-06-20 09:43:30.460 INFO  [27161425] [U4Random::flat@416]  m_seq_index    0 m_seq_nv  256 cursor    1 idx    1 d    0.43845


Problem with the backtrace. 

* no easy to automate way to see which process is doing this consumption (in debugger can find this by looking at fCurrentProcess in "f 4") 
* TODO: look at cfg4/CProcessManager probably can query Geant4 to get the relevant processes and their order when U4Random::flat gets called 
* could be unecessary sledgehammer as not many processes and probably the ordering can be discerned manually : so long as its consistent


