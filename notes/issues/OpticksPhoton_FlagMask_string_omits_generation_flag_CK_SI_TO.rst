OpticksPhoton_FlagMask_string_omits_generation_flag_CK_SI_TO
===============================================================

Overview
----------

* from ~/j/issues/junosw_offline_update_sept_2022.rst


TODO : Why is the CK origin flag not dumped ?
------------------------------------------------

Perhaps enum updates not accomodated ?::

    2022-09-30 00:03:43.810 DEBUG [162307] [junoSD_PMT_v2_Opticks::EndOfEvent@169] [ eventID 1 m_opticksMode 3
    2022-09-30 00:03:43.820 INFO  [162307] [junoSD_PMT_v2_Opticks::EndOfEvent@190]  eventID 1 num_hit 28 way_enabled 0
         0 gp.x     840.38 gp.y   19245.69 gp.z    1502.42 gp.R   19322.53 pmt    8114                SD|BT
         1 gp.x  -13430.50 gp.y   -7767.71 gp.z  -11408.74 gp.R   19258.11 pmt   14076             RE|SD|BT
         2 gp.x  -14922.10 gp.y   11530.04 gp.z    4201.58 gp.R   19320.04 pmt    6864             RE|SD|BT
         3 gp.x  -15151.85 gp.y    5609.26 gp.z   10628.09 gp.R   19339.04 pmt    3868             RE|SD|BT
         4 gp.x   10853.07 gp.y    1425.27 gp.z   15936.91 gp.R   19334.06 pmt    1390          RE|SC|SD|BT
         5 gp.x   11710.61 gp.y   14482.46 gp.z   -5129.43 gp.R   19318.16 pmt   11299                SD|BT
         6 gp.x  -17505.01 gp.y    6837.48 gp.z   -4430.66 gp.R   19308.22 pmt   10926             RE|SD|BT
         7 gp.x    5155.31 gp.y   17928.19 gp.z   -5137.70 gp.R   19349.24 pmt   11313          RE|SC|SD|BT
         8 gp.x   -4309.12 gp.y    6244.47 gp.z   17786.38 gp.R   19336.94 pmt     640             RE|SD|BT
         9 gp.x   18004.06 gp.y   -1181.39 gp.z    6883.82 gp.R   19311.37 pmt    5532             RE|SD|BT
        10 gp.x   -6796.09 gp.y  -17821.26 gp.z    2825.18 gp.R   19281.23 pmt    7360          RE|SC|SD|BT
        11 gp.x     402.42 gp.y   13634.73 gp.z   13677.00 gp.R   19316.53 pmt    2522                SD|BT

::

    232 #ifdef WITH_G4CXOPTICKS
    233     U4Hit hit ;
    234     U4HitExtra hit_extra ;
    235     U4HitExtra* hit_extra_ptr = way_enabled ? &hit_extra : nullptr ;
    236 
    237     for(int idx=0 ; idx < int(num_hit) ; idx++)
    238     {
    239         U4HitGet::FromEvt(hit, idx);
    240         collectHit(&hit, hit_extra_ptr, merged_count, savehit_count );
    241         if(idx < 20) dumpHit(idx, &hit, hit_extra_ptr );
    242     }


    411 #ifdef WITH_G4CXOPTICKS
    412 void junoSD_PMT_v2_Opticks::dumpHit(unsigned idx, const U4Hit* hit, const U4HitExtra* hit_extra ) const
    413 #elif WITH_G4OPTICKS
    414 void junoSD_PMT_v2_Opticks::dumpHit(unsigned idx, const G4OpticksHit* hit, const G4OpticksHitExtra* hit_extra ) const
    415 #endif
    416 {
    417     std::cout
    418         << std::setw(6) << idx
    419         << " gp.x " << std::setw(10) << std::fixed << std::setprecision(2) << hit->global_position.x()
    420         << " gp.y " << std::setw(10) << std::fixed << std::setprecision(2) << hit->global_position.y()
    421         << " gp.z " << std::setw(10) << std::fixed << std::setprecision(2) << hit->global_position.z()
    422         << " gp.R " << std::setw(10) << std::fixed << std::setprecision(2) << hit->global_position.mag()
    423         << " pmt "   << std::setw(7) << hit->sensor_identifier
    424         << " " << std::setw(20) << OpticksPhoton::FlagMask(hit->flag_mask, true)
    425         ;

    236 inline std::string OpticksPhoton::FlagMask(const unsigned mskhis, bool abbrev)
    237 {
    238     std::vector<const char*> labels ;
    239     unsigned lastBit = 17 ;
    240     assert( __MACHINERY == 0x1 << lastBit );
    241 
    242     for(unsigned n=0 ; n <= lastBit ; n++ )
    243     {
    244         unsigned flag = 0x1 << n ;
    245         if(mskhis & flag) labels.push_back( abbrev ? Abbrev(flag) : Flag(flag) );
    246     }
    247     unsigned nlab = labels.size() ;
    248 
    249     std::stringstream ss ;
    250     for(unsigned i=0 ; i < nlab ; i++ ) ss << labels[i] << ( i < nlab - 1 ? "|" : ""  ) ;
    251     return ss.str();
    252 }
    253 


::

    epsilon:sysrap blyth$ OpticksPhotonTest 
    2022-09-29 19:22:51.219 INFO  [16268535] [main@222]  sysrap.OpticksPhotonTest 
    2022-09-29 19:22:51.220 INFO  [16268535] [test_Abbrev_Flag@199] 
     n          0 (0x1 << n)          1 OpticksPhoton::Flag             CERENKOV OpticksPhoton::Abbrev                   CK
     n          1 (0x1 << n)          2 OpticksPhoton::Flag        SCINTILLATION OpticksPhoton::Abbrev                   SI
     n          2 (0x1 << n)          4 OpticksPhoton::Flag                 MISS OpticksPhoton::Abbrev                   MI
     n          3 (0x1 << n)          8 OpticksPhoton::Flag          BULK_ABSORB OpticksPhoton::Abbrev                   AB
     n          4 (0x1 << n)         16 OpticksPhoton::Flag          BULK_REEMIT OpticksPhoton::Abbrev                   RE
     n          5 (0x1 << n)         32 OpticksPhoton::Flag         BULK_SCATTER OpticksPhoton::Abbrev                   SC
     n          6 (0x1 << n)         64 OpticksPhoton::Flag       SURFACE_DETECT OpticksPhoton::Abbrev                   SD
     n          7 (0x1 << n)        128 OpticksPhoton::Flag       SURFACE_ABSORB OpticksPhoton::Abbrev                   SA
     n          8 (0x1 << n)        256 OpticksPhoton::Flag     SURFACE_DREFLECT OpticksPhoton::Abbrev                   DR
     n          9 (0x1 << n)        512 OpticksPhoton::Flag     SURFACE_SREFLECT OpticksPhoton::Abbrev                   SR
     n         10 (0x1 << n)       1024 OpticksPhoton::Flag     BOUNDARY_REFLECT OpticksPhoton::Abbrev                   BR
     n         11 (0x1 << n)       2048 OpticksPhoton::Flag    BOUNDARY_TRANSMIT OpticksPhoton::Abbrev                   BT
     n         12 (0x1 << n)       4096 OpticksPhoton::Flag                TORCH OpticksPhoton::Abbrev                   TO
     n         13 (0x1 << n)       8192 OpticksPhoton::Flag            NAN_ABORT OpticksPhoton::Abbrev                   NA
     n         14 (0x1 << n)      16384 OpticksPhoton::Flag      EFFICIENCY_CULL OpticksPhoton::Abbrev                   EX
     n         15 (0x1 << n)      32768 OpticksPhoton::Flag   EFFICIENCY_COLLECT OpticksPhoton::Abbrev                   EC
     n         16 (0x1 << n)      65536 OpticksPhoton::Flag             BAD_FLAG OpticksPhoton::Abbrev                   XX
     n         17 (0x1 << n)     131072 OpticksPhoton::Flag             BAD_FLAG OpticksPhoton::Abbrev                   XX
    epsilon:sysrap blyth$ 

Question becomes why::

     ((hit->flag_mask & (0x1 << 0)) == 0 )



::

     25 inline void U4HitGet::ConvertFromPhoton(U4Hit& hit,  const sphoton& global, const sphoton& local, const sphit& ht )
     26 {
     27     hit.zero();
     28 
     29     U4ThreeVector::FromFloat3( hit.global_position,      global.pos );
     30     U4ThreeVector::FromFloat3( hit.global_direction,     global.mom );
     31     U4ThreeVector::FromFloat3( hit.global_polarization,  global.pol );
     32 
     33     hit.time = double(global.time) ;
     34     hit.weight = 1. ;
     35     hit.wavelength = double(global.wavelength);
     36 
     37     U4ThreeVector::FromFloat3( hit.local_position,      local.pos );
     38     U4ThreeVector::FromFloat3( hit.local_direction,     local.mom );
     39     U4ThreeVector::FromFloat3( hit.local_polarization,  local.pol );
     40 
     41     hit.sensorIndex = ht.sensor_index ;
     42     hit.sensor_identifier = ht.sensor_identifier ;
     43     hit.nodeIndex = -1 ;
     44 
     45     hit.boundary = global.boundary() ;
     46     hit.photonIndex = global.idx() ;
     47     hit.flag_mask = global.flagmask ;
     48     hit.is_cerenkov = global.is_cerenkov() ;
     49     hit.is_reemission = global.is_reemit() ;
     50 }


     76 struct sphoton
     77 {
     78     float3 pos ;
     79     float  time ;
     80 
     81     float3 mom ;
     82     unsigned iindex ;  // instance index,  (formerly float weight, but have never used that)
     83 
     84     float3 pol ;
     85     float  wavelength ;
     86 
     87     unsigned boundary_flag ;
     88     unsigned identity ;
     89     unsigned orient_idx ;
     90     unsigned flagmask ;
     91 


::

    200 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    201 {
    202     sevent* evt      = params.evt ;
    203     if (launch_idx.x >= evt->num_photon) return;
    204 
    205     unsigned idx = launch_idx.x ;  // aka photon_idx
    206     unsigned genstep_idx = evt->seed[idx] ;
    207     const quad6& gs     = evt->genstep[genstep_idx] ;
    208 
    209     qsim* sim = params.sim ;
    210     curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 
    211 
    212     sctx ctx = {} ;
    213     ctx.evt = evt ;
    214     ctx.prd = prd ;
    215     ctx.idx = idx ;
    216 
    217     sim->generate_photon(ctx.p, rng, gs, idx, genstep_idx );
    218 

    1464 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    1465 {
    1466     const int& gencode = gs.q0.i.x ;
    1467     switch(gencode)
    1468     {
    1469         case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ;
    1470         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    1471 
    1472         case OpticksGenstep_G4Cerenkov_modified:
    1473         case OpticksGenstep_CERENKOV:
    1474                                               cerenkov->generate(    p, rng, gs, photon_id, genstep_id ) ; break ;
    1475 
    1476         case OpticksGenstep_DsG4Scintillation_r4695:
    1477         case OpticksGenstep_SCINTILLATION:
    1478                                               scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    1479 
    1480         case OpticksGenstep_INPUT_PHOTON:    { p = evt->photon[photon_id] ; p.set_flag(TORCH) ; }        ; break ;
    1481         default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ;
    1482     }
    1483 }



Added setting of initial sphoton::flagmask::

    epsilon:opticks blyth$ o
    On branch master
    Your branch is up-to-date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   qudarap/qcerenkov.h
        modified:   qudarap/qscint.h
        modified:   sysrap/OpticksPhoton.hh
        modified:   sysrap/tests/OpticksPhotonTest.cc




