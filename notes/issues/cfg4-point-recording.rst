cfg4-point-recording CRecorder::posttrackWritePoints experiment
==================================================================

Observations
---------------

* tis much simpler

  * actually the complexity is moved to collection, so not such a big win 

* perfect seqhis, see cfg4-bouncemax-not-working.rst
* seqmat totally messed up (no matswap)


* move to live style, messes up both... so hide behind *--recpoi* option

  * getting flag zeros, from Undefined boundary  
  * fix using inspiration from the dead code CRecorderLive.cc 
  * add setNote getNote to OpticksEvent to take note of recpoi or not (recstp) 


* comparing recpoi to recstp : get 3 seqhis/seqmat zeros from "TO AB" 

generate.cu : looks to mess up truncation too
--------------------------------------------------

* shift wrong 

::

    112 // beyond MAXREC overwrite save into top slot
    113 // TODO: check shift = slot_offset*4 rather than slot*4  ? 
    114 //       nope slot_offset*RNUMQUAD is absolute pointing into record_buffer
    115 //       so maybe
    116 //             shift = ( slot < MAXREC ? slot : MAXREC - 1 )* 4 
    117 //
    118 //       slot_offset constraint  
    119 //            int slot_min = photon_id*MAXREC ; 
    120 //            int slot_max = slot_min + MAXREC - 1 ;
    121 //            slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;
    122 //
    123 //        so in terms of saving location into record buffer, tis constrained correctly
    124 //        BUT the seqhis shifts look wrong in truncation 
    125 //
    126 
    127 #define RSAVE(seqhis, seqmat, p, s, slot, slot_offset)  \
    128 {    \
    129     unsigned int shift = slot*4 ; \
    130     unsigned long long his = __ffs((s).flag) & 0xF ; \
    131     unsigned long long mat = (s).index.x < 0xF ? (s).index.x : 0xF ; \
    132     seqhis |= his << shift ; \
    133     seqmat |= mat << shift ; \
    134     rsave((p), (s), record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain );  \
    135 }   \
    136 




FIXED : difference in topslot rewrite behaviour between recpoi and recstp
----------------------------------------------------------------------------

recpoi(CRec::addPoi)
   in case of truncation never reach lastPost so just::

       pre,pre,pre,pre,pre,...  

   until m_poi.size() >= m_ctx.point_limit() at which point returns 
   done=true and the track gets killed : topslot overwrite happens 
   in the CWriter if the points have been stored  
  

recstp(CRec::addStp)
    just adds steps until m_stp.size() >= m_ctx.step_limit() 
    but this limit is made large to prevent it having teeth : the 
    real truncation happening in CRecorder::posttrackWriteSteps

CWriter/CPhoton used by both these methods uses photon._slot_constrained
which will overwrite. But the writer cannot control the points that 
get given... so to align truncation the point_limit needs to 
match the truncation limit used in recstp/CWriter. 


::

     34 unsigned CG4Ctx::step_limit() const
     35 {
     36     // *step_limit* is used by CRec::addStp (recstp) the "canned" step collection approach, 
     37     // which just collects steps and makes sense of them later...
     38     // This has the disadvantage of needing to collect StepTooSmall steps (eg from BR turnaround)  
     39     // that are subsequently thrown : this results in the stem limit needing to 
     40     // be twice the size you might expect to handle hall-of-mirrors tboolean-truncate.
     41     assert( _ok_event_init );
     42     return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
     43 }
     44 
     45 unsigned CG4Ctx::point_limit() const
     46 {
     47     // *point_limit* is used by CRec::addPoi (recpoi) the "live" point collection approach, 
     48     // which makes sense of the points as they arrive, 
     49     // this has advantage of only storing the needed points. 
     50     //
     51     // DO NOT ADD +1 LEEWAY HERE : OTHERWISE TRUNCATION BEHAVIOUR IS CHANGED
     52     // see notes/issues/cfg4-point-recording.rst
     53     //
     54     assert( _ok_event_init );  
     55     return ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
     56 }




::

    167 void CWriter::writeStepPoint_(const G4StepPoint* point, const CPhoton& photon )
    168 {
    169     // write compressed record quads into buffer at location for the m_record_id 
    170 
    171     unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ;
    172     unsigned slot = photon._slot_constrained ;
    173     unsigned flag = photon._flag ;
    174     unsigned material = photon._mat ;
     
     



* do both recpoi+recstp, with writer disabled for one, and compare CPhoton

Of order 100/100000 have topslot rewrite difference::

    2017-11-18 15:14:48.527 INFO  [5630598] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-18 15:14:48.630 INFO  [5630598] [CRecorder::posttrack@112]  record_id 9290 event_id 0 track_id 9290 photon_id 9290 parent_id -1 primary_id -2 reemtrack 0
    2017-11-18 15:14:48.630 INFO  [5630598] [CRecorder::posttrack@113] ps:CPhoton slot_constrained 9 seqhis           6aaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:48.630 INFO  [5630598] [CRecorder::posttrack@114] pp:CPhoton slot_constrained 9 seqhis           aaaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:48.692 INFO  [5630598] [CRecorder::posttrack@112]  record_id 8810 event_id 0 track_id 8810 photon_id 8810 parent_id -1 primary_id -2 reemtrack 0
    2017-11-18 15:14:48.692 INFO  [5630598] [CRecorder::posttrack@113] ps:CPhoton slot_constrained 9 seqhis           aaaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:48.692 INFO  [5630598] [CRecorder::posttrack@114] pp:CPhoton slot_constrained 9 seqhis           6aaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:48.969 INFO  [5630598] [CRecorder::posttrack@112]  record_id 6621 event_id 0 track_id 6621 photon_id 6621 parent_id -1 primary_id -2 reemtrack 0
    2017-11-18 15:14:48.969 INFO  [5630598] [CRecorder::posttrack@113] ps:CPhoton slot_constrained 9 seqhis           aaaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:48.969 INFO  [5630598] [CRecorder::posttrack@114] pp:CPhoton slot_constrained 9 seqhis           6aaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:49.022 INFO  [5630598] [CRecorder::posttrack@112]  record_id 6201 event_id 0 track_id 6201 photon_id 6201 parent_id -1 primary_id -2 reemtrack 0
    2017-11-18 15:14:49.022 INFO  [5630598] [CRecorder::posttrack@113] ps:CPhoton slot_constrained 9 seqhis           6aaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:49.022 INFO  [5630598] [CRecorder::posttrack@114] pp:CPhoton slot_constrained 9 seqhis           aaaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:49.077 INFO  [5630598] [CRecorder::posttrack@112]  record_id 5764 event_id 0 track_id 5764 photon_id 5764 parent_id -1 primary_id -2 reemtrack 0
    2017-11-18 15:14:49.077 INFO  [5630598] [CRecorder::posttrack@113] ps:CPhoton slot_constrained 9 seqhis           6aaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:49.077 INFO  [5630598] [CRecorder::posttrack@114] pp:CPhoton slot_constrained 9 seqhis           aaaaaaaaad seqmat           2222222222 is_flag_done N is_done Y
    2017-11-18 15:14:49.121 INFO  [5630598] [CRecorder::posttrack@112]  record_id 5418 event_id 0 track_id 5418 photon_id 5418 parent_id -1 primary_id -2 reemtrack 0


    delta:optickscore blyth$ OpticksPhotonTest 
    2017-11-18 15:12:18.342 INFO  [5629618] [main@9] OpticksPhotonTest
     ( 0x1 <<  3 )  (i+1)  4 AB          BULK_ABSORB      8      8
     ( 0x1 <<  5 )  (i+1)  6 SC         BULK_SCATTER     20     32
     ( 0x1 <<  9 )  (i+1)  a SR     SURFACE_SREFLECT    200    512
     ( 0x1 << 10 )  (i+1)  b BR     BOUNDARY_REFLECT    400   1024
     ( 0x1 << 11 )  (i+1)  c BT    BOUNDARY_TRANSMIT    800   2048
     ( 0x1 << 12 )  (i+1)  d TO                TORCH   1000   4096
     ( 0x1 << 13 )  (i+1)  e NA            NAN_ABORT   2000   8192


::

    083 void CRecorder::posttrack() // invoked from CTrackingAction::PostUserTrackingAction
     84 {
     85     assert(!m_live);
     86 
     87     if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ;
     88 
     89     if(m_recpoi)
     90     {
     91 
     92 
     93         posttrackWritePoints();  // experimental alt 
     94 
     95         CPhoton pp(m_photon);
     96 
     97 
     98         m_writer->setEnabled(false);
     99 
    100         m_photon.clear();
    101         m_state.clear();
    102         posttrackWriteSteps();
    103 
    104         m_writer->setEnabled(true);
    105 
    106         CPhoton ps(m_photon);
    107 
    108 
    109 
    110         if(ps._seqhis != pp._seqhis)
    111         {
    112              LOG(info) << m_ctx.desc() ;
    113              LOG(info) << "ps:" << ps.desc() ;
    114              LOG(info) << "pp:" << pp.desc() ;
    115         }
    116 
    117     }
    118     else
    119     {
    120         posttrackWriteSteps();
    121     }
    122 
    123     if(m_dbg) m_dbg->posttrack();
    124 }





FIXED recpoi "TO AB" seqhis zero ISSUE 
---------------------------------------


::

    [2017-11-18 15:16:21,331] p96806 {/Users/blyth/opticks/ana/ab.py:146} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171118-1515 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171118-1515 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recpoi) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000        10.54/9 =  1.17  (pval:0.309 prob:0.691)  
    0000       aaaaaaaaad     99603     99586             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaaaaaa6ad        35        56             4.85        0.625 +- 0.106        1.600 +- 0.214  [10] TO SR SC SR SR SR SR SR SR SR
    0002       a6aaaaaaad        39        50             1.36        0.780 +- 0.125        1.282 +- 0.181  [10] TO SR SR SR SR SR SR SR SC SR
    0003       aaa6aaaaad        49        36             1.99        1.361 +- 0.194        0.735 +- 0.122  [10] TO SR SR SR SR SR SC SR SR SR
    0004       6aaaaaaaad        41        48             0.55        0.854 +- 0.133        1.171 +- 0.169  [10] TO SR SR SR SR SR SR SR SR SC
    0005       aaaaa6aaad        45        37             0.78        1.216 +- 0.181        0.822 +- 0.135  [10] TO SR SR SR SC SR SR SR SR SR
    0006       aaaaaa6aad        40        32             0.89        1.250 +- 0.198        0.800 +- 0.141  [10] TO SR SR SC SR SR SR SR SR SR
    0007       aaaa6aaaad        38        35             0.12        1.086 +- 0.176        0.921 +- 0.156  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        38             0.00        1.000 +- 0.162        1.000 +- 0.162  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        36             0.00        1.000 +- 0.167        1.000 +- 0.167  [10] TO SR SR SR SR SR SR SC SR SR
    0010             4aad         2        10             0.00        0.200 +- 0.141        5.000 +- 1.581  [4 ] TO SR SR AB
    0011         4aaaaaad         9         5             0.00        1.800 +- 0.600        0.556 +- 0.248  [8 ] TO SR SR SR SR SR SR AB
    0012               4d         4         9             0.00        0.444 +- 0.222        2.250 +- 0.750  [2 ] TO AB
    0013       4aaaaaaaad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [10] TO SR SR SR SR SR SR SR SR AB
    0014            4aaad         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [5 ] TO SR SR SR AB
    0015          4aaaaad         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [7 ] TO SR SR SR SR SR AB
    0016              4ad         4         1             0.00        4.000 +- 2.000        0.250 +- 0.250  [3 ] TO SR AB
    0017        4aaaaaaad         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [9 ] TO SR SR SR SR SR SR SR AB
    0018           4aaaad         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [6 ] TO SR SR SR SR AB
    .                             100000    100000        10.54/9 =  1.17  (pval:0.309 prob:0.691)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.43/2 =  0.22  (pval:0.806 prob:0.194)  
    0000             1200     99603     99586             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       368             0.07        0.981 +- 0.052        1.019 +- 0.053  [3 ] TO|SR|SC
    0002             1208        32        37             0.36        0.865 +- 0.153        1.156 +- 0.190  [3 ] TO|SR|AB
    0003             1008         4         9             0.00        0.444 +- 0.222        2.250 +- 0.750  [2 ] TO|AB
    .                             100000    100000         0.43/2 =  0.22  (pval:0.806 prob:0.194)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000       2222222222     99968     99960             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001             2222         2        10             0.00        0.200 +- 0.141        5.000 +- 1.581  [4 ] Vm Vm Vm Vm
    0002               22         4         9             0.00        0.444 +- 0.222        2.250 +- 0.750  [2 ] Vm Vm
    0003         22222222         9         5             0.00        1.800 +- 0.600        0.556 +- 0.248  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0004          2222222         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0005            22222         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [5 ] Vm Vm Vm Vm Vm
    0006              222         4         1             0.00        4.000 +- 2.000        0.250 +- 0.250  [3 ] Vm Vm Vm
    0007        222222222         2         3             0.00        0.667 +- 0.471        1.500 +- 0.866  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0008           222222         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [6 ] Vm Vm Vm Vm Vm Vm
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u




Only single point is collected
------------------------------------

::

    (lldb) p flag 
    (unsigned int) $0 = 4096
    (lldb) p boundary_status
    (DsG4OpBoundaryProcessStatus) $1 = StepTooSmall
    (lldb) p point
    (const G4StepPoint *) $2 = 0x00000001468130a0
    (lldb) p *point
    (const G4StepPoint) $3 = {
      fPosition = (dx = 115.23684692382813, dy = 8.2256317138671875, dz = -199.89999389648438)
      fGlobalTime = 0.20000000298023224
      fLocalTime = 0
      fProperTime = 0
      fMomentumDirection = (dx = -0, dy = -0, dz = 1)
      fKineticEnergy = 0.000003262741777421046
      fVelocity = 299.79244995117188
      fpTouchable = {
        fObj = 0x000000014680c810
      }
      fpMaterial = 0x000000010de01ce0
      fpMaterialCutsCouple = 0x000000010a07d880
      fpSensitiveDetector = 0x0000000000000000
      fSafety = 0
      fPolarization = (dx = 0, dy = -1, dz = 0)
      fStepStatus = fUndefined
      fpProcessDefinedStep = 0x0000000000000000
      fMass = 0
      fCharge = 0
      fMagneticMoment = 0
      fWeight = 1
    }
    (lldb) p num
    (unsigned int) $4 = 1
    (lldb) 


CPhoton dumps reveal getting (non-done) 0xd when should get 0x4d
------------------------------------------------------------------

::

    2017-11-18 11:04:21.800 INFO  [5554971] [CWriter::initEvent@80] CWriter::initEvent dynamic STATIC(GPU style) record_max 100000 bounce_max  9 steps_per_photon 10 num_g4event 10
    2017-11-18 11:04:22.086 INFO  [5554971] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-18 11:04:22.612 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 3 seqhis                 4aad seqmat                 2222 is_flag_done Y is_done Y
    2017-11-18 11:04:22.667 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 0 seqhis                    d seqmat                    2 is_flag_done N is_done N
    2017-11-18 11:04:22.851 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:23.046 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:04:23.123 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 0 seqhis                    d seqmat                    2 is_flag_done N is_done N
    2017-11-18 11:04:23.257 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:24.471 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 8 seqhis            4aaaaaaad seqmat            222222222 is_flag_done Y is_done Y
    2017-11-18 11:04:24.974 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:04:26.318 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:26.722 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 8 seqhis            4aaaaaaad seqmat            222222222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.217 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.522 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 4 seqhis                4aaad seqmat                22222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.647 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.709 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.725 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:04:27.995 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 0 seqhis                    d seqmat                    2 is_flag_done N is_done N
    2017-11-18 11:04:29.141 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 4 seqhis                4aaad seqmat                22222 is_flag_done Y is_done Y
    2017-11-18 11:04:29.234 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 3 seqhis                 4aad seqmat                 2222 is_flag_done Y is_done Y
    2017-11-18 11:04:30.389 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:30.525 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:04:32.207 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:04:32.365 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:04:32.557 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 5 seqhis               4aaaad seqmat               222222 is_flag_done Y is_done Y
    2017-11-18 11:04:33.626 INFO  [5554971] [CRecorder::posttrackWritePoints@219] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:04:34.140 INFO  [5554971] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2017-11-18 11:04:34.140 INFO  [5554971] [CG4::postpropagate@346] CG4::postpropagate(0)


::

    2017-11-18 11:07:21.019 INFO  [5556379] [CWriter::initEvent@80] CWriter::initEvent dynamic STATIC(GPU style) record_max 100000 bounce_max  9 steps_per_photon 10 num_g4event 10
    2017-11-18 11:07:21.300 INFO  [5556379] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-18 11:07:21.848 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 3 seqhis                 4aad seqmat                 2222 is_flag_done Y is_done Y
    2017-11-18 11:07:21.905 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 1 seqhis                   4d seqmat                   22 is_flag_done Y is_done Y
    2017-11-18 11:07:22.096 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:22.296 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:07:22.372 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 1 seqhis                   4d seqmat                   22 is_flag_done Y is_done Y
    2017-11-18 11:07:22.505 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:23.751 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 8 seqhis            4aaaaaaad seqmat            222222222 is_flag_done Y is_done Y
    2017-11-18 11:07:24.273 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:07:25.669 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:26.087 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 8 seqhis            4aaaaaaad seqmat            222222222 is_flag_done Y is_done Y
    2017-11-18 11:07:26.600 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:07:26.917 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 4 seqhis                4aaad seqmat                22222 is_flag_done Y is_done Y
    2017-11-18 11:07:27.047 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:07:27.111 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:27.128 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:07:27.405 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 1 seqhis                   4d seqmat                   22 is_flag_done Y is_done Y
    2017-11-18 11:07:28.588 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 4 seqhis                4aaad seqmat                22222 is_flag_done Y is_done Y
    2017-11-18 11:07:28.683 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 3 seqhis                 4aad seqmat                 2222 is_flag_done Y is_done Y
    2017-11-18 11:07:29.875 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:30.014 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:07:31.758 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 6 seqhis              4aaaaad seqmat              2222222 is_flag_done Y is_done Y
    2017-11-18 11:07:31.923 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 7 seqhis             4aaaaaad seqmat             22222222 is_flag_done Y is_done Y
    2017-11-18 11:07:32.122 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 5 seqhis               4aaaad seqmat               222222 is_flag_done Y is_done Y
    2017-11-18 11:07:33.228 INFO  [5556379] [CRecorder::posttrackWriteSteps@375] CPhoton slot_constrained 2 seqhis                  4ad seqmat                  222 is_flag_done Y is_done Y
    2017-11-18 11:07:33.759 INFO  [5556379] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2017-11-18 11:07:33.759 INFO  [5556379] [CG4::postpropagate@346] CG4::postpropagate(0)






::

   tboolean-;tboolean-truncate --okg4 --recpoi -D


    2017-11-17 20:58:53,847] p79170 {/Users/blyth/opticks/ana/seq.py:160} WARNING - SeqType.code check [?0?] bad 1 
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-2053 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171117-2053 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recpoi) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0003       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0004       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0012            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015               4d         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB

    0016                0         0         3             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ recpoi has 3 seqhis zeros, probably from "TO AB"

    0017        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0018             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0019           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    0000             1200     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       339             0.69        1.065 +- 0.056        0.939 +- 0.051  [3 ] TO|SR|SC
    0002             1208        32        25             0.86        1.280 +- 0.226        0.781 +- 0.156  [3 ] TO|SR|AB
    0003             1008         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|AB
    0004                0         0         3             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ]
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000       2222222222     99968     99976             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001         22222222         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0002              222         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] Vm Vm Vm
    0003          2222222         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0004            22222         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] Vm Vm Vm Vm Vm
    0005               22         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] Vm Vm
    0006                0         0         3             0.00        0.000 +- 0.000        0.000 +- 0.000  [1 ] ?0?
    0007        222222222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0008             2222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] Vm Vm Vm Vm
    0009           222222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm Vm Vm Vm
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 





    [2017-11-17 21:00:29,334] p79427 {/Users/blyth/opticks/ana/ab.py:146} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy () 
    B tboolean-truncate/torch/ -1 :  20171117-2100 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy (recstp) 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0003       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0004       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0012            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015               4d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO AB
    0016        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0018           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    0000             1200     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       339             0.69        1.065 +- 0.056        0.939 +- 0.051  [3 ] TO|SR|SC
    0002             1208        32        25             0.86        1.280 +- 0.226        0.781 +- 0.156  [3 ] TO|SR|AB
    0003             1008         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO|AB
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000       2222222222     99968     99976             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001         22222222         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0002              222         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] Vm Vm Vm
    0003          2222222         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0004            22222         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] Vm Vm Vm Vm Vm
    0005               22         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] Vm Vm
    0006        222222222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0007             2222         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] Vm Vm Vm Vm
    0008           222222         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] Vm Vm Vm Vm Vm Vm
    .                             100000    100000         0.00/0 =  0.00  (pval:nan prob:nan)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 







::

     77 void CRecorder::posttrack() // invoked from CTrackingAction::PostUserTrackingAction
     78 {
     79     assert(!m_live);
     80 
     81     if(m_ctx._dbgrec) LOG(info) << "CRecorder::posttrack" ;
     82 
     83     //posttrackWriteSteps();
     84     posttrackWritePoints();  // experimental alt 
     85 
     86     if(m_dbg) m_dbg->posttrack();
     87 }


::

    tboolean-truncate-p


    [2017-11-17 18:16:56,877] p65154 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-truncate)  None 0 
    A tboolean-truncate/torch/  1 :  20171117-1816 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/1/fdom.npy 
    B tboolean-truncate/torch/ -1 :  20171117-1816 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-truncate/torch/-1/fdom.npy 
    Rock//perfectSpecularSurface/Vacuum
    /tmp/blyth/opticks/tboolean-truncate--
    .                seqhis_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    0000       aaaaaaaaad     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [10] TO SR SR SR SR SR SR SR SR SR
    0001       aaa6aaaaad        49        42             0.54        1.167 +- 0.167        0.857 +- 0.132  [10] TO SR SR SR SR SR SC SR SR SR
    0002       6aaaaaaaad        41        49             0.71        0.837 +- 0.131        1.195 +- 0.171  [10] TO SR SR SR SR SR SR SR SR SC
    0003       aaaaa6aaad        45        42             0.10        1.071 +- 0.160        0.933 +- 0.144  [10] TO SR SR SR SC SR SR SR SR SR
    0004       aaaaaaa6ad        35        42             0.64        0.833 +- 0.141        1.200 +- 0.185  [10] TO SR SC SR SR SR SR SR SR SR
    0005       aaaaaa6aad        40        30             1.43        1.333 +- 0.211        0.750 +- 0.137  [10] TO SR SR SC SR SR SR SR SR SR
    0006       a6aaaaaaad        39        31             0.91        1.258 +- 0.201        0.795 +- 0.143  [10] TO SR SR SR SR SR SR SR SC SR
    0007       aaaa6aaaad        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SR SR SR SR SC SR SR SR SR
    0008       aaaaaaaa6d        38        36             0.05        1.056 +- 0.171        0.947 +- 0.158  [10] TO SC SR SR SR SR SR SR SR SR
    0009       aa6aaaaaad        36        31             0.37        1.161 +- 0.194        0.861 +- 0.155  [10] TO SR SR SR SR SR SR SC SR SR
    0010         4aaaaaad         9         4             0.00        2.250 +- 0.750        0.444 +- 0.222  [8 ] TO SR SR SR SR SR SR AB
    0011              4ad         4         6             0.00        0.667 +- 0.333        1.500 +- 0.612  [3 ] TO SR AB
    0012            4aaad         5         2             0.00        2.500 +- 1.118        0.400 +- 0.283  [5 ] TO SR SR SR AB
    0013          4aaaaad         5         4             0.00        1.250 +- 0.559        0.800 +- 0.400  [7 ] TO SR SR SR SR SR AB
    0014       4aaaaaaaad         4         4             0.00        1.000 +- 0.500        1.000 +- 0.500  [10] TO SR SR SR SR SR SR SR SR AB
    0015               4d         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO AB
    0016        4aaaaaaad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [9 ] TO SR SR SR SR SR SR SR AB
    0017             4aad         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO SR SR AB
    0018           4aaaad         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [6 ] TO SR SR SR SR AB
    .                             100000    100000         4.82/9 =  0.54  (pval:0.850 prob:0.150)  
    .                pflags_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    0000             1200     99603     99633             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SR
    0001             1220       361       339             0.69        1.065 +- 0.056        0.939 +- 0.051  [3 ] TO|SR|SC
    0002             1208        32        25             0.86        1.280 +- 0.226        0.781 +- 0.156  [3 ] TO|SR|AB
    0003             1008         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] TO|AB
    .                             100000    100000         1.56/2 =  0.78  (pval:0.459 prob:0.541)  
    .                seqmat_ana  1:tboolean-truncate   -1:tboolean-truncate        c2        ab        ba 
    .                             100000    100000    199914.00/9 = 22212.67  (pval:0.000 prob:1.000)  
    0000       2222222222     99968         0         99968.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Vm Vm Vm Vm Vm Vm Vm Vm Vm
    0001       1111111112         0     99633         99633.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Rk Rk
    0002       2111111112         0        53            53.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Rk Vm
    0003       1111111212         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Vm Rk Rk Rk Rk Rk Rk Rk
    0004       1112111112         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Vm Rk Rk Rk
    0005       1111121112         0        42            42.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Vm Rk Rk Rk Rk Rk
    0006       1111211112         0        36            36.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Vm Rk Rk Rk Rk
    0007       1111111122         0        36            36.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Vm Rk Rk Rk Rk Rk Rk Rk Rk
    0008       1211111112         0        31            31.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Rk Vm Rk
    0009       1121111112         0        31            31.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Rk Rk Rk Rk Vm Rk Rk
    0010       1111112112         0        30             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] Vm Rk Rk Vm Rk Rk Rk Rk Rk Rk
    0011         22222222         9         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Vm Vm Vm Vm Vm Vm Vm
    0012              212         0         6             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Rk Vm
    0013          2222222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm Vm Vm Vm Vm Vm
    0014            22222         5         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] Vm Vm Vm Vm Vm
    0015          2111112         0         4             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Rk Rk Rk Rk Rk Vm
    0016              222         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Vm Vm Vm
    0017         21111112         0         4             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm Rk Rk Rk Rk Rk Rk Vm
    0018               22         4         3             0.00        1.333 +- 0.667        0.750 +- 0.433  [2 ] Vm Vm
    0019             2112         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm Rk Rk Vm
    .                             100000    100000    199914.00/9 = 22212.67  (pval:0.000 prob:1.000)  
                /tmp/blyth/opticks/evt/tboolean-truncate/torch/1 7a4bcf2565d2235230cce18584128029 3c1a894417816154c638f8195e827bdc  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1}
    [2017-11-17 18:16:56,883] p65154 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:issues blyth$ 

