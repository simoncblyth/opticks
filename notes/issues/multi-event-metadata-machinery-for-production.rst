multi-event-metadata-machinery-for-production
===============================================

Overview
----------

Did this kinda thing many times, because metadata and layout keep changing
as well as the scope of the testing. What I want now : 

0. full SEvt loading is not needed, as too slow : want just the metadata
1. matrix of time stamps to see how reproducible the pattern is
2. summary plot presentation of the timing information  
3. once that is operational : start reduction from lots of debug to production minimalism

HMM: to create something that lasts could impl a nodata mode
for NP.hh and NPFold.h 

Priors
-------

ana/bench.py

ana/scan.py 

ana/profilesmryplot.py

ana/profilesmry.py

ana/profilesmrytab.py

ana/profile_.py

ana/metadata.py



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


Where is the equivalent for input photons ? 







