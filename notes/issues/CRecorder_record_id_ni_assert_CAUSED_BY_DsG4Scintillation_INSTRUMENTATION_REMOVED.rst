CRecorder_record_id_ni_assert_CAUSED_BY_DsG4Scintillation_INSTRUMENTATION_REMOVED
=====================================================================================

* unexpected extra photons (~250 in 100k) cause record_id asserts, switched to just logging them 
* non-instrumented reemission (ie reemission that does not pass along the CPhotonInfo lineage) could yield extras
* YEP : THATS THE CAUSE : MY INSTRUMENTATION HAS BEEN REMOVED



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


Is this an off by 1 with input photon record_id ? Nope, there are hundreds of extra photons.

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



Try to see what the extras are::


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
    412     _track_optical_count += 1 ;   // CAREFUL : DOES NOT ACCOUNT FOR RE-JOIN 
    413 
    414     assert( _record_id > -1 );
    415 
    416     
    417     if(_number_of_input_photons > 0 && _record_id > _number_of_input_photons)
    418     {   
    419         LOG(info)
    420             << " _number_of_input_photons " << _number_of_input_photons
    421             << " _photon_id " << _photon_id
    422             << " _record_id " << _record_id 
    423             << " _parent_id " << _parent_id
    424             << " _pho " << _pho.desc()
    425             << " _gs " << _gs.desc()
    426             << " _track_optical_count " << _track_optical_count
    427             ;
    428     }
    429     


::

    (gdb) p m_ctx._record_id
    $3 = 100000
    (gdb) p m_ctx._photon_id
    $4 = 100000
    (gdb) p m_ctx._pho
    $5 = {static MISSING = 4294967295, gs = 0, ix = 100000, id = 100000, gn = 0}
    (gdb) 


    (gdb) p m_ctx._track->GetTrackID()      ## HUH where did this extra track come from ?
    $7 = 100001
    (gdb) 

    ## photons getting into LS and reemitting creating secondaries should still have the primary record_id
 

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



Where do the extra 275 photons from event 0 come from ?::

    Begin of Event --> 0
    2021-06-18 18:39:26.625 INFO  [448794] [PMTEfficiencyCheck::addHitRecord@88]  m_eventID 0 m_record_count 0
    2021-06-18 18:39:26.625 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100000 m_ni 100000
    2021-06-18 18:39:26.628 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100001 m_ni 100000
    2021-06-18 18:39:26.641 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100002 m_ni 100000
    2021-06-18 18:39:26.649 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100003 m_ni 100000
    2021-06-18 18:39:26.670 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100004 m_ni 100000
    ...
    2021-06-18 18:39:28.972 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100273 m_ni 100000
    2021-06-18 18:39:28.972 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100274 m_ni 100000
    2021-06-18 18:39:28.978 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100275 m_ni 100000
    2021-06-18 18:39:29.006 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100276 m_ni 100000
    2021-06-18 18:39:29.020 INFO  [448794] [junoSD_PMT_v2_Opticks::EndOfEvent@128] [ eventID 0 m_opticksMode 3 numGensteps 1 numPhotons 100000

Extra 228 photons from event 1::

    2021-06-18 18:39:31.555 INFO  [448794] [G4Opticks::setInputPhotons@1934]  input_photons 1,4,4 repeat 100000
    Begin of Event --> 1
    2021-06-18 18:40:31.159 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100000 m_ni 100000
    2021-06-18 18:40:31.163 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100001 m_ni 100000
    2021-06-18 18:40:31.163 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100002 m_ni 100000
    2021-06-18 18:40:31.181 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100003 m_ni 100000
    2021-06-18 18:40:31.183 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100004 m_ni 100000
    ...
    2021-06-18 18:40:33.530 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100224 m_ni 100000
    2021-06-18 18:40:33.530 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100225 m_ni 100000
    2021-06-18 18:40:33.580 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100226 m_ni 100000
    2021-06-18 18:40:33.580 FATAL [448794] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100227 m_ni 100000
    2021-06-18 18:40:33.590 INFO  [448794] [junoSD_PMT_v2_Opticks::EndOfEvent@128] [ eventID 1 m_opticksMode 3 numGensteps 1 numPhotons 100000



Wind up the logging, points finger at reemission lineage failure::

    :set nowrap

    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2107 _record_id 2107 _parent_id -1 _pho CPho gs 0 ix 2107 id 2107 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98166
    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2106 _record_id 2106 _parent_id -1 _pho CPho gs 0 ix 2106 id 2106 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98167
    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2105 _record_id 2105 _parent_id -1 _pho CPho gs 0 ix 2105 id 2105 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98168
    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2104 _record_id 2104 _parent_id -1 _pho CPho gs 0 ix 2104 id 2104 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98169
    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2103 _record_id 2103 _parent_id -1 _pho CPho gs 0 ix 2103 id 2103 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98170
    2021-06-18 19:32:31.729 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2102 _record_id 2102 _parent_id -1 _pho CPho gs 0 ix 2102 id 2102 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98171
    2021-06-18 19:32:31.730 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 100273 _record_id 100273 _parent_id 2102 _pho CPho gs 0 ix 100273 id 100273 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98172
    2021-06-18 19:32:31.730 FATAL [68312] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100273 m_ni 100000
    2021-06-18 19:32:31.730 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 100274 _record_id 100274 _parent_id 100273 _pho CPho gs 0 ix 100274 id 100274 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98173
    2021-06-18 19:32:31.730 FATAL [68312] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100274 m_ni 100000
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2101 _record_id 2101 _parent_id -1 _pho CPho gs 0 ix 2101 id 2101 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98174
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2100 _record_id 2100 _parent_id -1 _pho CPho gs 0 ix 2100 id 2100 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98175
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2099 _record_id 2099 _parent_id -1 _pho CPho gs 0 ix 2099 id 2099 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98176
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2098 _record_id 2098 _parent_id -1 _pho CPho gs 0 ix 2098 id 2098 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98177
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2097 _record_id 2097 _parent_id -1 _pho CPho gs 0 ix 2097 id 2097 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98178
    2021-06-18 19:32:31.731 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 2096 _record_id 2096 _parent_id -1 _pho CPho gs 0 ix 2096 id 2096 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98179
    ...
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1835 _record_id 1835 _parent_id -1 _pho CPho gs 0 ix 1835 id 1835 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98440
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1834 _record_id 1834 _parent_id -1 _pho CPho gs 0 ix 1834 id 1834 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98441
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1833 _record_id 1833 _parent_id -1 _pho CPho gs 0 ix 1833 id 1833 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98442
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1832 _record_id 1832 _parent_id -1 _pho CPho gs 0 ix 1832 id 1832 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98443
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1831 _record_id 1831 _parent_id -1 _pho CPho gs 0 ix 1831 id 1831 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98444
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1830 _record_id 1830 _parent_id -1 _pho CPho gs 0 ix 1830 id 1830 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98445
    2021-06-18 19:32:31.758 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 100275 _record_id 100275 _parent_id 1830 _pho CPho gs 0 ix 100275 id 100275 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98446
    2021-06-18 19:32:31.759 FATAL [68312] [CWriter::writeStepPoint@206]  SKIP  unexpected record_id 100275 m_ni 100000
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1829 _record_id 1829 _parent_id -1 _pho CPho gs 0 ix 1829 id 1829 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98447
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1828 _record_id 1828 _parent_id -1 _pho CPho gs 0 ix 1828 id 1828 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98448
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1827 _record_id 1827 _parent_id -1 _pho CPho gs 0 ix 1827 id 1827 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98449
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1826 _record_id 1826 _parent_id -1 _pho CPho gs 0 ix 1826 id 1826 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98450
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1825 _record_id 1825 _parent_id -1 _pho CPho gs 0 ix 1825 id 1825 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98451
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1824 _record_id 1824 _parent_id -1 _pho CPho gs 0 ix 1824 id 1824 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98452
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1823 _record_id 1823 _parent_id -1 _pho CPho gs 0 ix 1823 id 1823 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98453
    2021-06-18 19:32:31.759 INFO  [68312] [CCtx::setTrackOptical@419]  _number_of_input_photons 100000 _photon_id 1822 _record_id 1822 _parent_id -1 _pho CPho gs 0 ix 1822 id 1822 gn 0 _gs T  idx   0 pho100000 off      0 _track_optical_count 98454






jcv DsPhysConsOptical::

    034 DsPhysConsOptical::DsPhysConsOptical(const G4String& name): G4VPhysicsConstructor(name)
     35                                                             , ToolBase(name)
     36 {
     37     declProp("OpticksMode", m_opticksMode=0);
     38     declProp("CerenMaxPhotonsPerStep", m_cerenMaxPhotonPerStep = 300);
     39     declProp("CerenPhotonStack", m_cerenPhotonStack = true);
     40 
     41     declProp("ScintDoReemission", m_doReemission = true);
     42     declProp("ScintDoScintAndCeren", m_doScintAndCeren = true);
     43     declProp("ScintDoReemissionOnly", m_doReemissionOnly = false);
     44 
     45     declProp("UseCerenkov", m_useCerenkov=true);
     46     declProp("UseCerenkovType", m_useCerenkovType="modified");
     47     declProp("ApplyWaterQe", m_applyWaterQe=true);
     48 
     49     declProp("UseScintillation", m_useScintillation=true);
     50     declProp("UseScintSimple", m_useScintSimple=false);
     51     declProp("UseRayleigh", m_useRayleigh=true);
     52     declProp("UseAbsorption", m_useAbsorption=true);
     53     declProp("UseAbsReemit", m_useAbsReemit=false);
     54     declProp("UseFastMu300nsTrick", m_useFastMu300nsTrick=false);
     55     declProp("ScintillationYieldFactor", m_ScintillationYieldFactor = 1.0);
     56   
    ...
    233 
    234      DsG4OpAbsReemit* absreemit_PPO =0;
    235      DsG4OpAbsReemit* absreemit_bisMSB =0;
    236       if (m_useAbsReemit){
    237                 absreemit_PPO= new DsG4OpAbsReemit("PPO");
    238                 absreemit_bisMSB= new DsG4OpAbsReemit("bisMSB");
    239                  absreemit_PPO->SetVerboseLevel(0);
    240                  absreemit_bisMSB->SetVerboseLevel(0);
    241               }
    242 
    243 
    244 
    245 
    246     G4OpAbsorption* absorb = 0;
    247     if (m_useAbsorption) {
    248         absorb = new G4OpAbsorption();
    249     }
    250 
    251     G4OpRayleigh* rayleigh = 0;
    252     if (m_useRayleigh) {
    253         rayleigh = new G4OpRayleigh();
    254     //        rayleigh->SetVerboseLevel(2);
    255     }


jcv JUNODetSimModule::

    0310         # add new optical model
     311 
     312         grp_pmt_op.add_argument("--new-optical-model", dest="new_optical_model", action="store_true",
     313                       help=mh("Use the new optical model."))
     314         grp_pmt_op.add_argument("--old-optical-model", dest="new_optical_model", action="store_false",
     315                       help=mh("Use the old optical model"))
     316         grp_pmt_op.set_defaults(new_optical_model=False)
     317 

    ...

    1454             op_process.property("UseQuenching").set(args.quenching)
    1455             # new optical model
    1456             if args.new_optical_model:
    1457                 op_process.property("UseAbsReemit").set(True)
    1458                 op_process.property("UseScintSimple").set(True)
    1459                 args.light_yield *= 0.9684
    1460             else:
    1461                 op_process.property("UseAbsReemit").set(False)
    1462                 op_process.property("UseScintSimple").set(False)
    1463             # pmt optical model
    1464             if args.pmt_optical_model:
    1465                 op_process.property("UsePMTOpticalModel").set(True)
    1466                 args.light_yield *= 0.8432
    1467             geom_info.property("LS.LightYield").set(args.light_yield)
    1468            # op_process.property("UseAbsReemit").set(args.absreemit)
    1469            # op_process.property("UseScintSimple").set(args.scintsimple)
    1470             # other flags:
    1471             op_process.property("doTrackSecondariesFirst").set(args.track_op_first)
        




My intrumentation has been removed::


    epsilon:offline blyth$ svn log  Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
    ------------------------------------------------------------------------
    r4690 | huyuxiang | 2021-06-18 07:39:49 +0100 (Fri, 18 Jun 2021) | 1 line

    redesign the scintillation time constant
    ------------------------------------------------------------------------
    r4677 | blyth | 2021-06-08 15:11:11 +0100 (Tue, 08 Jun 2021) | 1 line

    Opticks genstep collection update using CPhotonInfo to enable CRecorder/CWriter operation via G4OpticksRecorder in ordinary Opticks instrumented Geant4 running
    ------------------------------------------------------------------------
    r4497 | blyth | 2021-04-27 19:25:32 +0100 (Tue, 27 Apr 2021) | 1 line

    junoSD_PMT_v2::getMergerOpticks in opticksMode 1 return the standard m_pmthitmerger 
    ------------------------------------------------------------------------
    r4496 | blyth | 2021-04-26 15:06:22 +0100 (Mon, 26 Apr 2021) | 1 line

    switch to treating --opticksmode as bitfield, 0:Normal-Geant4-only 1:Opticks-only, plan: 2:Geant4-with-instrumentation plan: 3:Opticks-and-Geant4-with-comparison-machinery 
    ------------------------------------------------------------------------
    r4344 | huyuxiang | 2021-02-25 13:18:31 +0000 (Thu, 25 Feb 2021) | 1 line

    update the function of getting ScintillationYield 
    ------------------------------------------------------------------------
    r3987 | blyth | 2020-07-04 15:30:10 +0100 (Sat, 04 Jul 2020) | 1 line

    exclude collection of reemission gensteps




::


    epsilon:offline blyth$ svn diff -r 4677:4690 Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
    Index: Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
    ===================================================================
    --- Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc	(revision 4677)
    +++ Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc	(revision 4690)
    @@ -276,36 +276,7 @@
         G4String strYieldRatio = "YIELDRATIO";
     
         // reset the slower time constant and ratio
    -    slowerTimeConstant = 0.0;
    -    slowerRatio = 0.0;
    -    
    -    if (aParticleName == "opticalphoton") {
    -      FastTimeConstant = "ReemissionFASTTIMECONSTANT";
    -      SlowTimeConstant = "ReemissionSLOWTIMECONSTANT";
    -      strYieldRatio = "ReemissionYIELDRATIO";
    -    }
    -    else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
    -      FastTimeConstant = "GammaFASTTIMECONSTANT";
    -      SlowTimeConstant = "GammaSLOWTIMECONSTANT";
    -      strYieldRatio = "GammaYIELDRATIO";
    -      slowerTimeConstant = gammaSlowerTime;
    -      slowerRatio = gammaSlowerRatio;
    -    }
    -    else if(aParticleName == "alpha") {
    -      FastTimeConstant = "AlphaFASTTIMECONSTANT";
    -      SlowTimeConstant = "AlphaSLOWTIMECONSTANT";
    -      strYieldRatio = "AlphaYIELDRATIO";
    -      slowerTimeConstant = alphaSlowerTime;
    -      slowerRatio = alphaSlowerRatio;
    -    }
    -    else {
    -      FastTimeConstant = "NeutronFASTTIMECONSTANT";
    -      SlowTimeConstant = "NeutronSLOWTIMECONSTANT";
    -      strYieldRatio = "NeutronYIELDRATIO";
    -      slowerTimeConstant = neutronSlowerTime;
    -      slowerRatio = neutronSlowerRatio;
    -    }
    -
    +   
         const G4MaterialPropertyVector* Fast_Intensity = 
             aMaterialPropertiesTable->GetProperty("FASTCOMPONENT"); 
         const G4MaterialPropertyVector* Slow_Intensity =
    @@ -319,11 +290,23 @@
         if (!Fast_Intensity && !Slow_Intensity )
             return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
     
    -    G4int nscnt = 1;
    -    if (Fast_Intensity && Slow_Intensity) nscnt = 2;
    -    if ( verboseLevel > 0) {
    -      G4cout << " Fast_Intensity " << Fast_Intensity << " Slow_Intensity " << Slow_Intensity << " nscnt " << nscnt << G4endl;
    +    //-------------find the type of particle------------------------------//
    +    G4MaterialPropertyVector* Ratio_timeconstant = 0 ;
    +    if (aParticleName == "opticalphoton") {
    +      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
         }
    +    else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
    +      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("GammaCONSTANT");
    +    }
    +    else if(aParticleName == "alpha") {
    +      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("AlphaCONSTANT");
    +    }
    +    else {
    +      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("NeutronCONSTANT");
    +    }
    +    
    +  //-----------------------------------------------------//
    +
         G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
         G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
     
    @@ -330,6 +313,7 @@
         G4ThreeVector x0 = pPreStepPoint->GetPosition();
         G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
         G4double      t0 = pPreStepPoint->GetGlobalTime();
    +     
     
         //Replace NumPhotons by NumTracks
         G4int NumTracks=0;
    @@ -456,6 +440,7 @@
               G4cout << " Generated " << NumTracks << " scint photons. mean(scint photons) = " << MeanNumberOfTracks << G4endl;
             }
         }
    +
         weight*=fPhotonWeight;
         if ( verboseLevel > 0 ) {
           G4cout << " set scint photon weight to " << weight << " after multiplying original weight by fPhotonWeight " << fPhotonWeight 
    @@ -491,188 +476,76 @@
         // new G4PhysicsOrderedFreeVector allocated to hold CII's
     
         G4int Num = NumTracks; //# tracks is now the loop control
    -        
    -    G4double fastTimeConstant = 0.0;
    -    if (flagDecayTimeFast) { // Fast Time Constant
    -        const G4MaterialPropertyVector* ptable =
    -        aMaterialPropertiesTable->GetProperty(FastTimeConstant.c_str());
    -        if (verboseLevel > 0) {
    -          G4cout << " MaterialPropertyVector table " << ptable << " for FASTTIMECONSTANT"<<G4endl;
    -        }
    -        if (!ptable) ptable = aMaterialPropertiesTable->GetProperty("FASTTIMECONSTANT");
    -        if (ptable) {
    -            fastTimeConstant = ptable->Value(0);
    -          if (verboseLevel > 0) { 
    -            G4cout << " dump fast time constant table " << G4endl;
    -            ptable->DumpValues();
    -          }
    -        }
    -    }
     
    -    G4double slowTimeConstant = 0.0;
    -    if (flagDecayTimeSlow) { // Slow Time Constant
    -        const G4MaterialPropertyVector* ptable =
    -        aMaterialPropertiesTable->GetProperty(SlowTimeConstant.c_str());
    -        if (verboseLevel > 0) {
    -          G4cout << " MaterialPropertyVector table " << ptable << " for SLOWTIMECONSTANT"<<G4endl;
    -        }
    -        if(!ptable) ptable = aMaterialPropertiesTable->GetProperty("SLOWTIMECONSTANT");
    -        if (ptable){
    -          slowTimeConstant = ptable->Value(0);
    -          if (verboseLevel > 0) { 
    -            G4cout << " dump slow time constant table " << G4endl;
    -            ptable->DumpValues();
    -          }
    -        }
    -    }
    -
    -    G4double YieldRatio = 0.0;
    -    { // Slow Time Constant
    -        const G4MaterialPropertyVector* ptable =
    -            aMaterialPropertiesTable->GetProperty(strYieldRatio.c_str());
    -        if(!ptable) ptable = aMaterialPropertiesTable->GetProperty("YIELDRATIO");
    -        if (ptable) {
    -            YieldRatio = ptable->Value(0);
    -            if (verboseLevel > 0) {
    -                G4cout << " YieldRatio = "<< YieldRatio << " and dump yield ratio table (yield ratio = fast/(fast+slow): " << G4endl;
    -                (ptable)->DumpValues();
    +   
    +    size_t nscnt = Ratio_timeconstant->GetVectorLength();
    +    std::vector<G4int> m_Num(nscnt);
    +    m_Num.clear();
    +    for(G4int i = 0 ; i < NumTracks ; i++){
    +       G4double p = G4UniformRand();
    +       G4double p_count = 0;
    +       for(G4int j = 0 ; j < nscnt; j++)
    +       {
    +            p_count += (*Ratio_timeconstant)[j] ;
    +            if( p < p_count ){
    +               m_Num[j]++ ;
    +               break;
                 }
    -        }
    -    }
    +        }  
    +  
    +     }
     
     
    -    //loop over fast/slow scintillations
    -    for (G4int scnt = 1; scnt <= nscnt; scnt++) {
    +    for(G4int scnt = 0 ; scnt < nscnt ; scnt++){
     
    -        G4double ScintillationTime = 0.*ns;
    -        G4PhysicsOrderedFreeVector* ScintillationIntegral = NULL;
    +         G4double ScintillationTime = 0.*ns;
    +         G4PhysicsOrderedFreeVector* ScintillationIntegral = NULL;
     
    -        if (scnt == 1) {//fast
    -            if (nscnt == 1) {
    -                if(Fast_Intensity){
    -                    ScintillationTime   = fastTimeConstant;
    -                    ScintillationIntegral =
    -                        (G4PhysicsOrderedFreeVector*)((*theFastIntegralTable)(materialIndex));
    -                }
    -                if(Slow_Intensity){
    -                    ScintillationTime   = slowTimeConstant;
    -                    ScintillationIntegral =
    -                        (G4PhysicsOrderedFreeVector*)((*theSlowIntegralTable)(materialIndex));
    -                }
    -            }
    -            else {
    -                if ( ExcitationRatio == 1.0 ) {
    -                  Num = G4int( 0.5 +  (min(YieldRatio,1.0) * NumTracks) );  // round off, not truncation
    -                }
    -                else {
    -                  Num = G4int( 0.5 +  (min(ExcitationRatio,1.0) * NumTracks));
    -                }
    -                if ( verboseLevel>1 ){
    -                  G4cout << "Generate Num " << Num << " optical photons with fast component using NumTracks " 
    -                         << NumTracks << " YieldRatio " << YieldRatio << " ExcitationRatio " << ExcitationRatio 
    -                         << " min(YieldRatio,1.)*NumTracks = " <<  min(YieldRatio,1.)*NumTracks 
    -                         << " min(ExcitationRatio,1.)*NumTracks = " <<  min(ExcitationRatio,1.)*NumTracks 
    -                         << G4endl;
    -                }
    -                ScintillationTime   = fastTimeConstant;
    -                ScintillationIntegral =
    +         if ( scnt == 0 ){
    +              ScintillationIntegral =
                         (G4PhysicsOrderedFreeVector*)((*theFastIntegralTable)(materialIndex));
    -            }
    -        }
    -        else {//slow
    -            Num = NumTracks - Num;
    -            ScintillationTime   =   slowTimeConstant;
    -            ScintillationIntegral =
    -                (G4PhysicsOrderedFreeVector*)((*theSlowIntegralTable)(materialIndex));
    -        }
    -        if (verboseLevel > 0) {
    -          G4cout << "generate " << Num << " optical photons with scintTime " << ScintillationTime 
    -                 << " slowTimeConstant " << slowTimeConstant << " fastTimeConstant " << fastTimeConstant << G4endl;
    -        }
    +         }
    +         else{
    +              ScintillationIntegral =
    +                    (G4PhysicsOrderedFreeVector*)((*theSlowIntegralTable)(materialIndex));
    +         }         
    +         
    +       //  G4int m_Num =G4int(NumTracks * (*Ratio_timeconstant)[scnt]);
    +         ScintillationTime = Ratio_timeconstant->Energy(scnt);
    +         if (!flagDecayTimeFast && scnt == 0){
    +               ScintillationTime = 0.*ns  ;
    +         }
     
    -        if (!ScintillationIntegral) continue;
    -        
    -        // Max Scintillation Integral
    +         if (!flagDecayTimeSlow && scnt != 0){
     
    -#ifdef WITH_G4OPTICKS
    -        
    -        /**
    -        Opticks Collection of scintillation gensteps prior to generation loop
    -        -----------------------------------------------------------------------
    +               ScintillationTime = 0.*ns  ;
    +         }
     
    -        ancestor_id:-1 
    -           normal case, meaning that aTrack was not a photon
    -           so the generation loop will yield "primary" photons 
    -           with id : gs.offset + i  
    -        
    -        ancestor_id>-1
    -           aTrack is a photon that may be about to reemit (Num=0 or 1) 
    -           ancestor_id is the absolute id of the primary parent photon, 
    -           this id is retained thru any subsequent remission secondary generations
     
    -        **/
    -        bool fabricate_unlabelled = false ; 
    -        CPho ancestor = CPhotonInfo::Get(&aTrack, fabricate_unlabelled); 
    -        int ancestor_id = ancestor.get_id() ; 
    -        if(ancestor_id > -1 ) assert( Num == 0 || Num == 1);  
    -
    -        CGenstep gs ; 
    -
    -        bool is_opticks_genstep = Num > 0 && !flagReemission ; 
    -
    -        if(is_opticks_genstep && (m_opticksMode & 1))
    -        {
    -            gs = G4Opticks::Get()->collectGenstep_DsG4Scintillation_r3971(
    -                &aTrack,
    -                &aStep,
    -                Num,
    -                scnt,
    -                slowerRatio,
    -                slowTimeConstant,
    -                slowerTimeConstant,
    -                ScintillationTime
    -            );
    -        }
    -#endif
    -
    -        if( m_opticksMode == 0 || (m_opticksMode & 2) )
    -        {
    - 
    -        for (G4int i = 0; i < Num; i++) { 
    -
    -#ifdef WITH_G4OPTICKS
    -         unsigned photon_id = ancestor_id > -1 ? ancestor_id : gs.offset + i ; 
    -         G4Opticks::Get()->setAlignIndex(photon_id); 
    -#endif
    -
    -        if(scnt == 2) {
    -            ScintillationTime = slowTimeConstant;
    -            if(flagDecayTimeSlow && G4UniformRand() < slowerRatio && (!flagReemission)) ScintillationTime = slowerTimeConstant;
    -        }
    -
    -            G4double sampledEnergy;
    -            if ( !flagReemission ) {
    +         for(G4int i = 0 ; i < m_Num[scnt] ; i++) {
    +           G4double sampledEnergy;
    +           if ( !flagReemission ) {
                     // normal scintillation
    -                G4double CIIvalue = G4UniformRand()*
    +               G4double CIIvalue = G4UniformRand()*
                         ScintillationIntegral->GetMaxValue();
    -                sampledEnergy=
    +               sampledEnergy=
                         ScintillationIntegral->GetEnergy(CIIvalue);
     
    -                if (verboseLevel>1) 
    +               if (verboseLevel>1) 
                         {
                             G4cout << "sampledEnergy = " << sampledEnergy << G4endl;
                             G4cout << "CIIvalue =        " << CIIvalue << G4endl;
                         }
                 }
    -            else {
    +         else {
                     // reemission, the sample method need modification
                     G4double CIIvalue = G4UniformRand()*
                         ScintillationIntegral->GetMaxValue();
                     if (CIIvalue == 0.0) {
    -                    // return unchanged particle and no secondaries  
    +                    // return unchanged particle and no secondaries 
                         aParticleChange.SetNumberOfSecondaries(0);
                         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    -                }
    +                   }
                     sampledEnergy=
                         ScintillationIntegral->GetEnergy(CIIvalue);
                     if (verboseLevel>1) {
    @@ -679,11 +552,10 @@
                         G4cout << "oldEnergy = " <<aTrack.GetKineticEnergy() << G4endl;
                         G4cout << "reemittedSampledEnergy = " << sampledEnergy
                                << "\nreemittedCIIvalue =        " << CIIvalue << G4endl;
    -                }
    -            }
    -
    -            // Generate random photon direction
    -
    +                   }
    +             }
    +        
    +           // Generate random photon direction
                 G4double cost = 1. - 2.*G4UniformRand();
                 G4double sint = sqrt((1.-cost)*(1.+cost));
     
    @@ -695,12 +567,10 @@
                 G4double py = sint*sinp;
                 G4double pz = cost;
     
    -            // Create photon momentum direction vector 
    -
    +            // Create photon momentum direction vector  
                 G4ParticleMomentum photonMomentum(px, py, pz);
     
                 // Determine polarization of new photon 
    -
                 G4double sx = cost*cosp;
                 G4double sy = cost*sinp; 
                 G4double sz = -sint;
    @@ -717,8 +587,8 @@
     
                 photonPolarization = photonPolarization.unit();
     
    -            // Generate a new photon:
    -
    +            // Generate a new photon:    
    +        
                 G4DynamicParticle* aScintillationPhoton =
                     new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(), 
                                           photonMomentum);
    @@ -730,7 +600,6 @@
                 aScintillationPhoton->SetKineticEnergy(sampledEnergy);
     
                 // Generate new G4Track object:
    -
                 G4double rand=0;
                 G4ThreeVector aSecondaryPosition;
                 G4double deltaTime;
    @@ -774,25 +643,9 @@
                 G4Track* aSecondaryTrack = 
                     new G4Track(aScintillationPhoton,aSecondaryTime,aSecondaryPosition);
     
    -            //G4CompositeTrackInfo* comp=new G4CompositeTrackInfo();
    -            //DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
    -            //if ( flagReemission ){
    -            //    if ( reemittedTI ) *trackinf = *reemittedTI;
    -            //    trackinf->SetReemitted();
    -            //}
    -            //else if ( fApplyPreQE ) {
    -            //    trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
    -            //    trackinf->SetQE(fPreQE);
    -            //}
    -            //comp->SetPhotonTrackInfo(trackinf);
    -            //aSecondaryTrack->SetUserInformation(comp);
    -                
                 aSecondaryTrack->SetWeight( weight );
                 aSecondaryTrack->SetTouchableHandle(aStep.GetPreStepPoint()->GetTouchableHandle());
    -            // aSecondaryTrack->SetTouchableHandle((G4VTouchable*)0);//this is wrong
    -                
                 aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    -                
                 // add the secondary to the ParticleChange object
                 aParticleChange.SetSecondaryWeightByProcess( true ); // recommended
                 aParticleChange.AddSecondary(aSecondaryTrack);
    @@ -799,28 +652,13 @@
                     
                 aSecondaryTrack->SetWeight( weight );
                 if ( verboseLevel > 0 ) {
    -              G4cout << " aSecondaryTrack->SetWeight( " << weight<< " ) ; aSecondaryTrack->GetWeight() = " << aSecondaryTrack->GetWeight() << G4endl;}
    -            // The above line is necessary because AddSecondary() 
    -            // overrides our setting of the secondary track weight, 
    -            // in Geant4.3.1 & earlier. (and also later, at least 
    -            // until Geant4.7 (and beyond?)
    -            //  -- maybe not required if SetWeightByProcess(true) called,
    -            //  but we do both, just to be sure)
    +              G4cout << " aSecondaryTrack->SetWeight( " << weight<< " ) ; aSecondaryTrack->GetWeight() = " << aSecondaryTrack->GetWeight() << G4endl;}        
    +         }    
    +   
    +   }
     
     
    -#ifdef WITH_G4OPTICKS
    -            aSecondaryTrack->SetUserInformation(CPhotonInfo::MakeScintillation(gs, i, ancestor ));
    -            G4Opticks::Get()->setAlignIndex(-1);
    -#endif
     
    -        }   // end loop over Num secondary photons
    -
    -        }   // (opticksMode == 0) || (opticksMode & 2 )   : opticks not enabled, or opticks enabled and doing Geant4 comparison
    -
    -
    -
    -    } // end loop over fast/slow scints
    -
         if (verboseLevel > 0) {
             G4cout << "\n Exiting from G4Scintillation::DoIt -- NumberOfSecondaries = " 
                    << aParticleChange.GetNumberOfSecondaries() << G4endl;
    epsilon:offline blyth$ 



