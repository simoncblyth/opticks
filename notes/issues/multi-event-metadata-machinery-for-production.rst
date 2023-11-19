multi-event-metadata-machinery-for-production
===============================================


Priors
-------

ana/bench.py

ana/scan.py 

ana/profilesmryplot.py

ana/profilesmry.py

ana/profilesmrytab.py

ana/profile_.py

ana/metadata.py


Overview
----------

Did this kinda thing many times, because metadata and layout keep changing
as well as the scope of the testing. What I want now : 

0. full SEvt loading is not needed, as too slow : want just the metadata
1. matrix of time stamps to see how reproducible the pattern is
2. summary plot presentation of the timing information  
3. once that is operational : start reduction from lots of debug to production minimalism
4. perhaps a timestamp mode that makes A and B easily comparable, but arranging 
   the same stamp names ?  

HMM: to create something that lasts could impl a nodata mode
for NP.hh and NPFold.h 

Can do most of the hardlifting in C++ writing a summary NPFold 
that can read from python and present with matplotlib

Status
---------

Investigating in::

    ~/np/tests/NPFold_nodata_test.sh
    ~/np/tests/NPFold_stamps_test.sh

1. DONE : added nodata mode to NP.hh and NPFold.h : for fast loading of 
          metadata only from many folders of arrays 
2. DONE : added timestamp metadata handling to NP.hh and NPFold.h
3. DONE : in NPFold_stamps_test.sh NPFold::substamps_pn
          select subfold of an NPFold to be compared based on prefix
          asserting on same counts pull the timestamp info into a collective table to be
          saved with names and metadata  (hmm as python is the 
          consumer could create char arrays for names of timestamps
          and filepaths)

4. TODO : plotting timestamps 

* maybe can use things from::

    sysrap/sevt_tt.py 
    j/ntds/stamp.sh 




HMM : not just timestamps, also need memory record
-----------------------------------------------------

OLD OK_PROFILE approach from Opticks.hh treated the code site label as dynamic, 
and collected it into a char array.

* approach is rather limited : profiling info collected into 
  the global Opticks instance

* collecting into SEvt NPFold makes more sense 

::

    76 #define OK_PROFILE(s) \
     77     { \
     78        if(m_ok)\
     79        {\
     80           m_ok->profile((s)) ;\
     81        }\
     82     }
     83 

     558 void Opticks::profile(const char* label)
     559 {
     560     if(!m_profile_enabled) return ;
     561     m_profile->stamp(label, m_tagoffset);
     562    // m_tagoffset is set by Opticks::makeEvent
     563 }

::

     366 Opticks::Opticks(int argc, char** argv, const char* argforced )
     367     :
     ...
     382     m_profile(new OpticksProfile()),
     383     m_profile_enabled(m_sargs->hasArg("--profile")),

::

    ./optickscore/OpticksEvent.cc:    OK_PROFILE("_OpticksEvent::setGenstepData");
    ./optickscore/OpticksEvent.cc:    OK_PROFILE("OpticksEvent::setGenstepData");

    209 void OpticksProfile::stamp(const char* label, int count)
    210 {
    211    setT(BTimeStamp::RealTime2()) ;
    212    setVM(SProc::VirtualMemoryUsageMB()) ;
    213    m_num_stamp += 1 ;
    214 
    215    float  t   = m_t - m_t0 ;      // time since instanciation
    216    float dt   = m_t - m_tprev ;   // time since previous stamp
    217 
    218    float vm   = m_vm - m_vm0 ;     // vm since instanciation
    219    float dvm  = m_vm - m_vmprev ;  // vm since previous stamp
    220 
    221    m_last_stamp.x = t ;
    222    m_last_stamp.y = dt ;
    223    m_last_stamp.z = vm ;
    224    m_last_stamp.w = dvm ;
    225 
    226    // the prev start at zero, so first dt and dvm give absolute m_t0 m_vm0 valules
    227 
    228    m_tt->add<const char*>(label, t, dt, vm, dvm,  count );
    229    m_npy->add(       t, dt, vm, dvm );
    230    m_lpy->addString( label ) ;

    m_npy(NPY<float>::make(0,m_tt->getNumColumns())),
    m_lpy(NPY<char>::make(0,64)),



NP.hh NPFold.h nodata mode : load just metadata from NPFold tree of arrays
-----------------------------------------------------------------------------

FIXED Issue 1 : reading NPFold from the run folder trips up on run_meta.txt : by skipping sidecars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Issue 2 : NPFold.h repetion of hit.npy::

    n010
    NPFold::desc depth 1
     loaddir:/hpcfs/juno/junogpu/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/n010 subfold 0 ff 0 kk 21 aa 21
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                           hit.npy : (3735, 4, 4, )
                       genstep.npy : (1, 6, 4, )
                        photon.npy : (10000, 4, 4, )
                        record.npy : (10000, 32, 4, 4, )
                           seq.npy : (10000, 2, 2, )
                           prd.npy : (10000, 32, 2, 4, )
                           hit.npy : (3735, 4, 4, )
                        domain.npy : (2, 4, 4, )
                      inphoton.npy : (10000, 4, 4, )
                           tag.npy : (10000, 4, )
                          flat.npy : (10000, 64, )
                           aux.npy : (10000, 32, 4, 4, )
                           sup.npy : (10000, 6, 4, )


HMM : the index has the repetition so problem on saving not loading::

    epsilon:n010 blyth$ cat NPFold_index.txt
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    hit.npy
    genstep.npy
    photon.npy
    record.npy
    seq.npy
    prd.npy
    hit.npy
    domain.npy
    inphoton.npy
    tag.npy
    flat.npy
    aux.npy
    sup.npy
    epsilon:n010 blyth$ 


HMM : maybe related to clear_except("hit") ? But why Geant4 event only ?::

    1247 void SEvt::endOfEvent(int eventID)
    1248 {   
    1249     int index_ = 1+eventID ;    
    1250     endIndex(index_);   // also sets t_EndOfEvent stamp
    1251     
    ...
    1268     setMeta<double>("t_Launch", t_Launch ); 
    1270     
    1271     save();              // gather and save SEventConfig configured arrays
    1272     clear_except("hit"); 
    1273     // an earlier SEvt::clear is invoked by QEvent::setGenstep before launch 
    1274 
    1275 }

    1448 void SEvt::clear_except(const char* keep)
    1449 {
    1450     LOG(LEVEL) << "[" ;
    1451     clear_vectors();
    1452 
    1453     bool copy = false ;
    1454     char delim = ',' ;
    1455     if(fold) fold->clear_except(keep, copy, delim);
    1456 
    1457     LOG(LEVEL) << "]" ;
    1458 }


U4Recorder must be doing smth different in its SEvt handling vs QEvent ?

TODO: SEvt/NPFold lifecycle tests::

     gather_components 
     clear
     clear_except 

Mockup how NPFold is being reused for each event 
with components coming and going. 


HMM : genstep handling with input photons ? 
---------------------------------------------


::

    167     quad6 gs_ = MakeGenstep_DsG4Scintillation_r4695( aTrack, aStep, numPhotons, scnt, ScintillationTime);
    168 
    169 #ifdef WITH_CUSTOM4
    170     sgs _gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    171     gs = C4GS::Make(_gs.index, _gs.photons, _gs.offset, _gs.gentype );
    172 #else
    173     gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    174 #endif
    175     // gs is private static genstep label 
    176 


    281 #ifdef WITH_CUSTOM4
    282     sgs _gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    283     gs = C4GS::Make(_gs.index, _gs.photons, _gs.offset , _gs.gentype );
    284 #else
    285     gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    286 #endif


With scintillation and cerenkov U4.cc adds the genstep to both SEvt::EGPU and SEvt::ECPU 
via the static::

    1030 sgs SEvt::AddGenstep(const quad6& q)
    1031 {
    1032     sgs label = {} ;
    1033     if(Exists(0)) label = Get(0)->addGenstep(q) ;
    1034     if(Exists(1)) label = Get(1)->addGenstep(q) ;
    1035     return label ;
    1036 }


Where is the equivalent for input photons ? Its done from SEvt::addFrameGenstep. 







