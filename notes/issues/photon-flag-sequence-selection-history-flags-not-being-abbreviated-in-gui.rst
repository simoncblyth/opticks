photon-flag-sequence-selection-history-flags-not-being-abbreviated-in-gui
==============================================================================

::

   OKTest --target 62590 --xanalytic --eye -0.9,0,0 --generateoverride -1

   OpticksIdx=error OKTest --target 62590 --xanalytic --eye -0.9,0,0 --generateoverride -1
         ## turn on dumping of the seqhis index table 

   OpticksIdx=error OKTest --target 62590 --xanalytic --eye -0.9,0,0 --generateoverride -1 --save 






* OGLRap/Photons m_seqhis(GItemList)

::

    413 void OpticksViz::indexPresentationPrep()
    414 {
    415     if(!m_idx) return ;
    416 
    417     LOG(LEVEL) ;
    418 
    419     m_seqhis = m_idx->makeHistoryItemIndex();
    420     m_seqmat = m_idx->makeMaterialItemIndex();
    421     m_boundaries = m_idx->makeBoundaryItemIndex();
    422 
    423 }


::

     50 OKMgr::OKMgr(int argc, char** argv, const char* argforced )
     51     :
     52     m_log(new SLog("OKMgr::OKMgr","", debug)),
     53     m_ok(Opticks::HasInstance() ? Opticks::GetInstance() : new Opticks(argc, argv, argforced)),
     54     m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
     55     m_idx(new OpticksIdx(m_hub)),
     56     m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
     57     m_gen(m_hub->getGen()),
     58     m_run(m_hub->getRun()),
     59     m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
     60     m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
     61     m_count(0)
     62 {
     63     init();
     64     (*m_log)("DONE");
     65 }


::

     76 GItemIndex* OpticksIdx::makeHistoryItemIndex()
     77 {
     78     OpticksEvent* evt = getEvent();
     79     Index* seqhis_ = evt->getHistoryIndex() ;
     80     if(!seqhis_)
     81     {    
     82          LOG(warning) << "OpticksIdx::makeHistoryItemIndex NULL seqhis" ;
     83          return NULL ; 
     84     }
     85 
     86     OpticksAttrSeq* qflg = m_hub->getFlagNames();
     87     //qflg->dumpTable(seqhis, "OpticksIdx::makeHistoryItemIndex seqhis"); 
     88 
     89     GItemIndex* seqhis = new GItemIndex(seqhis_) ; 
     90     seqhis->setTitle("Photon Flag Sequence Selection");
     91     seqhis->setHandler(qflg);
     92     seqhis->formTable();
     93 
     94     return seqhis ;
     95 }
     96 



::

    2019-10-13 16:09:29.031 INFO  [136268] [OpticksAttrSeq::dumpTable@332] OpticksIdx::makeHistoryItemIndex seqhis
        0    263895     0.317           8cccccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
        1    130076     0.156          8ccccc6dTORCH BULK_SCATTER BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
        2    116464     0.140                4d                                TORCH BULK_ABSORB 
        3     53584     0.064         8ccccc66dTORCH BULK_SCATTER BULK_SCATTER BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
        4     45725     0.055               46d                   TORCH BULK_SCATTER BULK_ABSORB 
        5     29629     0.036          4ccccccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BULK_ABSORB 
        6     20647     0.025        8ccccc666dTORCH BULK_SCATTER BULK_SCATTER BULK_SCATTER BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
        7     17239     0.021              466d      TORCH BULK_SCATTER BULK_SCATTER BULK_ABSORB 
        8     13912     0.017          8ccccc5dTORCH BULK_REEMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
        9     11313     0.014            8ccccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB 
       10     11100     0.013               4cd              TORCH BOUNDARY_TRANSMIT BULK_ABSORB 
       11      9370     0.011            4ccccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BULK_ABSORB 
       12      8981     0.011               45d                    TORCH BULK_REEMIT BULK_ABSORB 
       13      8583     0.010        ccccc6666dTORCH BULK_SCATTER BULK_SCATTER BULK_SCATTER BULK_SCATTER BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT 
       14      7056     0.008        cccccbcccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_REFLECT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT 
       15      6718     0.008              4c6d TORCH BULK_SCATTER BOUNDARY_TRANSMIT BULK_ABSORB 
       16      6718     0.008              4ccdTORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT BULK_ABSORB 


::

    305 std::string OpticksAttrSeq::getLabel(Index* index, const char* key, unsigned int& colorcode)
    306 {
    307     colorcode = 0xFFFFFF ;
    308 
    309     unsigned int source = index->getIndexSource(key); // the count for seqmat, seqhis
    310     float fraction      = index->getIndexSourceFraction(key);
    311     std::string dseq    = m_ctrl & HEXKEY
    312                                 ?
    313                                    decodeHexSequenceString(key)
    314                                 :
    315                                    decodeString(key)
    316                                 ;
    317 
    318     std::stringstream ss ;
    319     ss
    320         << std::setw(10) << source
    321         << std::setw(10) << std::setprecision(3) << std::fixed << fraction
    322         << std::setw(18) << ( key ? key : "-" )
    323         << std::setw(50) << dseq
    324         ;
    325 
    326     return ss.str();
    327 }


::

    [blyth@localhost opticks]$ opticks-f savehit
    ./bin/scan.bash:   local cmd="ts $(scan-ph-lv) --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --production --savehit --multievent 10 --xanalytic "  ; 
    ./bin/scan.bash:   local cmd="ts $(scan-px-lv) --oktest --pfx $(scan-pfx) --cat ${cat}_${num_abbrev} --generateoverride ${num_photons} --compute --production --savehit --multievent 10 --xanalytic "  ; 
    ./optickscore/OpticksEvent.cc:        if(m_ok->hasOpt("savehit")) saveHitData();  // FOR production hit check
    ./optickscore/OpticksCfg.cc:       ("savehit",   "save hits even in production running") ; 
    [blyth@localhost opticks]$ 


::

    161 /**
    162 TBuf::downloadSelection4x4
    163 -----------------------------
    164 
    165 This hides the float4x4 type down in here in the _.cu 
    166 where nvcc makes it available by default, such that the  
    167 user doesnt need access to the type.
    168 
    169 **/
    170 
    171 unsigned TBuf::downloadSelection4x4(const char* name, NPY<float>* npy, unsigned hitmask, bool verbose) const
    172 {
    173     return downloadSelection<float4x4>(name, npy, hitmask, verbose);
    174 }


    [blyth@localhost opticks]$ opticks-f downloadSelection4x4
    ./optixrap/OEvent.cc:    unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    ./optixrap/OEvent.cc:    unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    ./thrustrap/tests/TBuf4x4Test.cu:    tpho.downloadSelection4x4("tpho.downloadSelection4x4", hit, hitmask );
    ./thrustrap/TBuf_.cu:TBuf::downloadSelection4x4
    ./thrustrap/TBuf_.cu:unsigned TBuf::downloadSelection4x4(const char* name, NPY<float>* npy, unsigned hitmask, bool verbose) const 
    ./thrustrap/TBuf.hh:      unsigned downloadSelection4x4(const char* name, NPY<float>* npy, unsigned mskhis, bool verbose=false) const ; // selection done on items of size float4x4
    ./okop/tests/compactionTest.cc:    LOG(error) << "[ tpho.downloadSelection4x4 "; 
    ./okop/tests/compactionTest.cc:    tpho.downloadSelection4x4("thit<float4x4>", hit, mskhis, verbose );
    ./okop/tests/compactionTest.cc:    LOG(error) << "] tpho.downloadSelection4x4 "; 
    [blyth@localhost opticks]$ 



::

    [blyth@localhost torch]$ np.py 1/ht.npy 
    a :                                                     1/ht.npy :       (239820, 4, 4) : 28292e1615e48297d1954b76b181ad97 : 20191013-1647 


