Multi Event
=============

FIXED : Issue
--------------

Multi event test causes hard CUDA crash on launch of 2nd event forcing reboot...

::
   
    OKTest --compute --multievent 3

    pIndexer::indexSequenceCompute@214: OpIndexer::indexSequenceCompute
    OpticksIdx::indexBoundariesHost@211: OpticksIdx::indexBoundariesHost dpho NULL or no data 
    OKMgr::propagate@81: OKMgr::propagate DONE 1
    OKMgr::propagate@76: OKMgr::propagate 2
    *OpticksHub::initOKEvent@609: OpticksHub::initOKEvent  gs 1,6,4
    OpticksHub::setGensteps@274: OpticksHub::setGensteps shape 1,6,4 oac : GS_FABRICATED GS_TORCH 
    OpticksHub::setGensteps@301:  checklabel of torch steps  oac : GS_FABRICATED GS_TORCH 
    G4StepNPY::checklabel@170: G4StepNPY::checklabel xlabel 4096 ylabel -1
    G4StepNPY::Summary@136: OpticksHub::setGensteps  TORCH: 4096 CERENKOV: 1 SCINTILLATION: 2 G4GUN: 16384
    OpticksEvent::~OpticksEvent@151: OpticksEvent::~OpticksEvent PLACEHOLDER
    *NPY<float>::load@307: NPY<T>::load failed for path [/tmp/blyth/opticks/evt/dayabay/torch/1/1_track.npy] use debugload to see why 
    OpticksEvent::resize@709: OpticksEvent::resize  num_photons 100000 num_records 1000000 maxrec 10
    *OpticksHub::initOKEvent@630: OpticksHub::initOKEvent  gensteps 1,6,4 tagdir /tmp/blyth/opticks/evt/dayabay/torch/1
    OPropagator::initEvent@212: OPropagator::initEvent count 1 size(100000,1)
    OPropagator::updateEventBuffers@239: OPropagator::updateEventBuffers  EXPERIMENT IN REUSE OF OPTIX BUFFERS 
    OContext::configureBuffer@428:   gensteps               1,6,4 QUAD size (gnq)          6   
    OContext::upload@287: OContext::upload numBytes 96upload (1,6,4)  NumBytes(0) 96 NumBytes(1) 96 NumValues(0) 24 NumValues(1) 24{}
    OpticksEvent::checkData@636:  setting buffer ctrl  name photon dctrl 0 :  sctrl 136 : OPTIX_INPUT_OUTPUT PTR_FROM_OPENGL 
    OContext::configureBuffer@428:     photon          100000,4,4 QUAD size (gnq)     400000
    OpticksEvent::checkData@636:  setting buffer ctrl  name record dctrl 0 :  sctrl 32 : OPTIX_OUTPUT_ONLY 
    OContext::configureBuffer@428:         rx       100000,10,2,4 QUAD size (gnq)    2000000
    OpticksEvent::checkData@636:  setting buffer ctrl  name sequence dctrl 0 :  sctrl 36 : OPTIX_NON_INTEROP OPTIX_OUTPUT_ONLY 
    OContext::configureBuffer@420:         sq          100000,1,2 USER size (ijk)     200000 elementsize 8
    OpSeeder::seedPhotonsFromGensteps@61: OpSeeder::seedPhotonsFromGensteps
    OpSeeder::seedPhotonsFromGenstepsImp@148: OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    OpZeroer::zeroRecords@54: OpZeroer::zeroRecords
    OContext::launch@220: OContext::launch entry 0 width 0 height 0
    OContext::launch@220: OContext::launch entry 0 width 100000 height 1


FIX
----

Fixed by a major reworking of buffer handling, including adoption of new WITH_SEED_BUFFER approach.


Workaround
------------

Using the normal `OPropagator::initEventBuffers` for every event 
rather than `OPropagator::updateEventBuffers` avoids the issue, but that 
means are recreating OptiX buffers for every event.


Things to try
---------------

* Does buffer recreation always mean a long prelaunch ?
* DONE: isolated test of changing OptiX buffer, filling with some marker stripes, using the same machinery 
* DONE : Try skipping prelaunch, for events after the 1st  : makes no difference


DONE : bufferTest.cu bufferTest.cc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This indicates that can resize OptiX buffers between launches without issue. 
Also can start with a zero sized buffer. 

Used this pattern in OEvent OPropagator using m_zero OpticksEvent.


But this has issue in interop as to craete the OpenGL buffers
will need to OpticksViz::uploadEvent() for this zero event.


Trivial Program works
~~~~~~~~~~~~~~~~~~~~~~~~

::

    OKTest --compute --trivial --multievent 2 


::

    simon:cu blyth$ cat ../numquad.h 
    #define GNUMQUAD 6  // quads per genstep  
    #define PNUMQUAD 4  // quads per photon  
    #define RNUMQUAD 2  // quads per record  


But the NUMQUAD defines coming out with wrong values::

    (trivial) photon_id 59 photon_offset 0 genstep_id 236 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 60 photon_offset 0 genstep_id 240 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 61 photon_offset 0 genstep_id 244 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 62 photon_offset 0 genstep_id 248 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 63 photon_offset 0 genstep_id 252 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 32 photon_offset 0 genstep_id 128 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 33 photon_offset 0 genstep_id 132 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 
    (trivial) photon_id 34 photon_offset 0 genstep_id 136 GNUMQUAD 0 PNUMQUAD 6 RNUMQUAD 4 genstep_offset 2 

That turns out to be a new rtPrintf buf in OptiX 400.




