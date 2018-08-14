G4OK_propagation_matching
===========================

How to proceed ?
-------------------

1. bring in the big guns : CFG4.CRecorder instruments the Geant4 propagation
   enabling step-by-step recording of photons in OpticksEvent format : for 
   direct comparison with the OpticksEvent from the Opticks GPU propagation

   * this aint so easy, CRecorder was setup to operate with CG4 
   * will need to factor out the essential parts of the CRecorder and 
     make them more generally applicable 
   * start by reviewing/documenting CFG4 focussing on CRecorder 

2. work on aligning Cerenkov generation, get aligned mode to operate 
   within the direct approach 

   * detailed recording will help with this


Unfortunately a side effect of both the above 
is that they will complicate the hell out of the example. 

* leave CerenkovMinimal (ckm-) as is and start new example  CerenkovInstrumented (cki-) ?


Loading ckm geocache for propagation viz
------------------------------------------

::

    ckm-load()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKTest --load --natural --envkey
        type $FUNCNAME
    }
    ckm-dump()
    {
        OPTICKS_KEY=$(ckm-key) OpticksEventDumpTest --natural --envkey
        type $FUNCNAME
    }


Actually easier to bring direct geometry into CFG4 that vice-versa
---------------------------------------------------------------------

This is especially so as have a geocache of the direct geometry : so 
this means little new development, can just try to get something like
the below to work::

    ckm-cfg4()
    {   
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey
    }



Actually need soon to test many example direct geometries, so follow tboolean pattern
---------------------------------------------------------------------------------------

Expand on the tboolean pattern of double executable, one to apply the direct conversion 
and write the geometry (actually this first executable can be a simple example) 
and the second executable to run with it.
tboolean- only did that with test geometries, but theres no reason not to do it 
will "full though small" geocache geometries. The second executable can then be the 
fully CFG4 instrumented "--okg4" OKG4Test gorilla. 

Hmm : that will convert the geocache loaded geometry back into an Geant4 geometry !
 

::

    epsilon:~ blyth$ t tboolean-box
    tboolean-box is a function
    tboolean-box () 
    { 
        TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $*
    }
    epsilon:~ blyth$ 


::

    172 op-binary-name()
    173 {
    174    case $1 in
    175          --version) echo OpticksCMakeConfigTest ;;
    176          --idpath) echo OpticksIDPATH ;;
    177            --keys) echo InteractorKeys ;;
    178           --tcfg4) echo CG4Test ;;
    179            --okg4) echo OKG4Test ;;
    180            --okx4) echo OKX4Test ;;
    181          --tracer) echo OTracerTest ;;


