CRecorder_record_id_ni_assert
==============================



::

    unotoptask:DetSimAlg.execute   INFO: DetSimAlg Simulate An Event (0) 
    junoSD_PMT_v2::Initialize
    2021-06-18 18:11:16.147 INFO  [407989] [junoSD_PMT_v2_Opticks::Initialize@84]  tool 0x1e37490 input_photons 0x2fe5f40 input_photon_repeat 100000 g4ok 0x4cdddf0
    2021-06-18 18:11:16.147 INFO  [407989] [G4Opticks::setInputPhotons@1934]  input_photons 1,4,4 repeat 100000
    Begin of Event --> 0
    2021-06-18 18:12:14.031 INFO  [407989] [PMTEfficiencyCheck::addHitRecord@88]  m_eventID 0 m_record_count 0
    python: /home/blyth/opticks/cfg4/CWriter.cc:204: bool CWriter::writeStepPoint(const G4StepPoint*, unsigned int, unsigned int, bool): Assertion `record_id < m_ni' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #3  0x00007ffff6cf2252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffcdd84a9a in CWriter::writeStepPoint (this=0x153addac0, point=0x24ce700, flag=4096, material=1, last=false) at /home/blyth/opticks/cfg4/CWriter.cc:204
    #5  0x00007fffcdd7bca2 in CRecorder::WriteStepPoint (this=0x153add940, point=0x24ce700, flag=4096, material=1, boundary_status=Undefined, last=false) at /home/blyth/opticks/cfg4/CRecorder.cc:754
    #6  0x00007fffcdd7b268 in CRecorder::postTrackWriteSteps (this=0x153add940) at /home/blyth/opticks/cfg4/CRecorder.cc:644
    #7  0x00007fffcdd7962f in CRecorder::postTrack (this=0x153add940) at /home/blyth/opticks/cfg4/CRecorder.cc:212
    #8  0x00007fffcdda6468 in CManager::postTrack (this=0x153add680) at /home/blyth/opticks/cfg4/CManager.cc:346


    (gdb) f 4
    #4  0x00007fffcdd84a9a in CWriter::writeStepPoint (this=0x153addac0, point=0x24ce700, flag=4096, material=1, last=false) at /home/blyth/opticks/cfg4/CWriter.cc:204
    204	    assert( record_id < m_ni ); 
    (gdb) p record_id
    $1 = 100000
    (gdb) p m_ni
    $2 = 100000
    (gdb) 


Is this an off by 1 with input photon record_id ?

::

    194 bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last )
    195 {
    196     unsigned record_id = m_ctx._record_id ;
    197 
    198     LOG(LEVEL)
    199         << " m_ctx._photon_id " << m_ctx._photon_id
    200         << " m_ctx._record_id " << m_ctx._record_id
    201         << " m_ni " << m_ni
    202         ;
    203 
    204     assert( record_id < m_ni );
    205     assert( m_records_buffer );
    206 
    207 
    208     m_photon.add(flag, material);  // sets seqhis/seqmat nibbles in current constrained slot  
    209 
    210     bool hard_truncate = m_photon.is_hard_truncate();



    395 void CCtx::setTrackOptical(G4Track* mtrack)
    396 {
    397     mtrack->UseGivenVelocity(true);
    398 
    399     bool fabricate_unlabelled = true ;
    400     _pho = CPhotonInfo::Get(mtrack, fabricate_unlabelled);
    401 
    402     int pho_id = _pho.get_id();
    403     assert( pho_id > -1 );
    404 
    405     _gs = _gsc->getGenstep(_pho.gs) ;
    406     assert( _gs.index == _pho.gs );
    407 
    408     _photon_id = pho_id ; // 0-based, absolute photon index within the event 
    409     _record_id = pho_id ; // used by CRecorder/CWriter is now absolute, following abandonment of onestep mode  
    410     _record_fraction = double(_record_id)/double(_record_max) ;
    411 


::

    (gdb) p m_ctx._record_id
    $3 = 100000
    (gdb) p m_ctx._photon_id
    $4 = 100000
    (gdb) p m_ctx._pho
    $5 = {static MISSING = 4294967295, gs = 0, ix = 100000, id = 100000, gn = 0}
    (gdb) 


    (gdb) p m_ctx._track->GetTrackID()      ## HUH where did this come from ?
    $7 = 100001
    (gdb) 

    (gdb) p m_ctx._pdg_encoding 
    $8 = 20022



::

    194 bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last )
    195 {   
    196     unsigned record_id = m_ctx._record_id ;  
    197 
    198     LOG(LEVEL)  
    199         << " m_ctx._photon_id " << m_ctx._photon_id 
    200         << " m_ctx._record_id " << m_ctx._record_id
    201         << " m_ni " << m_ni
    202         ;
    203     
    204     if( record_id >= m_ni )
    205     {
    206         LOG(fatal) 
    207             << " SKIP "
    208             << " unexpected record_id " << record_id
    209             << " m_ni " << m_ni
    210             ;
    211         return ; 
    212     }
    213     //assert( record_id < m_ni ); 
    214     
    215     assert( m_records_buffer );

