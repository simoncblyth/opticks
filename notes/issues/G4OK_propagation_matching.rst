G4OK_propagation_matching
===========================

How to proceed ? How to apply CFG4 with directly converted geometry ? 
-------------------------------------------------------------------------

1. bring in the big guns : CFG4.CRecorder instruments the Geant4 propagation
   enabling step-by-step recording of photons in OpticksEvent format : for 
   direct comparison with the OpticksEvent from the Opticks GPU propagation

   * this aint so easy, CRecorder was setup to operate with CG4 
   * will need to factor out the essential parts of the CRecorder and 
     make them more generally applicable 
   * start by reviewing/documenting CFG4 focussing on CRecorder 

2. work on aligning Cerenkov generation, get aligned mode to operate 
   within the direct approach 

   * detailed recording will help with this, start with restricting 
     bouncemax to zero : to compare generated photons


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



Actually need to test many example direct geometries, so follow tboolean pattern
---------------------------------------------------------------------------------------


tboolean pattern expanded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expand on the tboolean pattern of two executables:

1. applies the direct conversion and writes the geometry geocache 
   with GDML included. Actually this first executable can be a simple example, 
   that also does an Opticks and/or G4 propagation : but without 
   all the instrumentation complications.   

2. second executable to run with the geometry, which 
   can be the fully CFG4 instrumented "--okg4" OKG4Test gorilla
   
::

    epsilon:~ blyth$ t tboolean-box
    tboolean-box is a function
    tboolean-box () 
    { 
        TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $*
    }
    epsilon:~ blyth$ 


tboolean- only did that with test geometries, but theres no reason not to do it 
will "full though small" geocache geometries. 

Hmm : that will convert the geocache loaded geometry back into an Geant4 geometry !
The convert back to G4 functionality in CTestDetector is limited to simple
russian doll geometries : for general case need to use CGDMLDetector which 
will need to be used together with the geocache, as GDML drops some info.

So this gets back to the okg4/tests/OKX4Test but with a variety of 
OPTICKS_KEY geocached with GDML rather than always wih the standard DYB.
So how to make that work.

1. construct g4 geometry in code (or generate it when scanning over all solids)
2. apply direct conversion X4PhysicalVolume to create GGeo, 
3. persist G4 geometry to GDML in the geocache  
4. persist GGeo to geocache 
  

OKG4Test : idea is to this as a "2nd" executable to do the instrumented bi-simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_KEY=$(ckm-key) OKG4Test --compute --envkey 


* key geocache with the GDML (as part of geocache) succeed to recover the original geometry for both simulations
* BUT: the source of photons is not recovered 

  * in original ckm- that source is some gensteps that are collected , Opticks in original or
    2nd executable is happy to propagate those... (if persist them somewhere appropriate : 
    probably inside an evt folder inside the live geocache makes sense)

  * BUT Cerenkov doesnt have the capability to replay some gensteps,
    could it be hacked to do that ?

  * hmm a way around this is to use inputphotons from an emitsource : 
    which both Opticks and G4 can use (this is what worked before, in the aligned test)


Notice there is a potential problem with inconsistency of the original physics (Cerenkov) 
implementation in 1st (example) executable and in the 2nd (coming from CFG4).  Could just use the 
CFG4.Cerenkov in the example ... but thats complicating the example.  The idea is to 
keep the 1st executable simple and do all complex instrumentation and trickery in the 2nd 
executable only.  

To this end giving CFG4.Cerenkov capability to replay canned gensteps fits in, 
although is means that the photons will be in places out of step with the other particles, unless 
succeed to arrange a reproducible rest of the event. 
But they dont matter really : the objective is to compare the photon propagation between the 
two simulations. True but, hacking Cerekov replaying is too whacky and niche : 
better to develop code that makes more general sense and thus has more use than the immediate problem, ie

* parametrize and persist the gun source (there is something like that already : perhaps in opposite direction
  converting a commandline gun specification into a G4 gun) : need the opposite get the spec from the gun 
* verify that can use it to reproducibly get the same non-opticals in 1st and 2nd executables, then should
  have reproducible G4 cerenkov photons (and gensteps) : will need to use same physics to do this  







Argh GDML matrix values truncation again : this is just cause to update G4 : DONE : UPDATED TO 10.4.2
--------------------------------------------------------------------------------------------------------

* :doc:`GDML_matrix_values_truncation`

::

     4   <define>
      5     <matrix coldim="2" name="EFFICIENCY0x10e2939d0" values="2.034e-06 0.5 4.136e-06 0.5"/>
      6     <matrix coldim="2" name="RINDEX0x10e2933c0" values="2.034e-06 1.49 4.136e-06 1.49"/>
      7     <matrix coldim="2" name="RINDEX0x10e292390" values="2.034e-06 1.3435 2.068e-06 1.344 2.103e-06 1.3445 2.139e-06 1.345 2.177e-06 1.3455 2.216e-06 1.346 2"/>
      8     <matrix coldim="2" name="RINDEX0x10e2906c0" values="2.034e-06 1 2.068e-06 1 2.103e-06 1 2.139e-06 1 2.177e-06 1 2.216e-06 1 2.256e-06 1 2.298e-06 1 2.34"/>
      9   </define>











