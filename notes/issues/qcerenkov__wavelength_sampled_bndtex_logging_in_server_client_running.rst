qcerenkov__wavelength_sampled_bndtex_logging_in_server_client_running
========================================================================

Overview : This issue reveals need for big QCerenkov cleanup
--------------------------------------------------------------

* issue is not server-client related : get same problem in mono-running

* WIP : QCerenkov split off old ICDF impl behind QCERENKOV_ICDF_OLD
* WIP : revive + rework QCerenkovTest.sh for current qcerenkov.h bnd based impl

  * need to test what are actually using


HMM: looking like all photons from all CK gensteps afflicted ?
-----------------------------------------------------------------

::


          INFO   127.0.0.1:35796 - "POST /simulate HTTP/1.1" 200
    //qcerenkov::wavelength_sampled_bndtex idx   4387 sampledRI   1.263 cosTheta   1.195 sin2Theta   0.000 wavelength 179.010 count 100 gsid 58 gs.matline 39 gs.pos (-134.874, 79.721,202.973)
    //qcerenkov::wavelength_sampled_bndtex idx   1927 sampledRI   1.263 cosTheta   1.219 sin2Theta   0.000 wavelength 204.238 count 100 gsid 36 gs.matline 39 gs.pos (-142.892, 81.467,202.901)
    //qcerenkov::wavelength_sampled_bndtex idx   1928 sampledRI   1.263 cosTheta   1.219 sin2Theta   0.000 wavelength 382.924 count 100 gsid 36 gs.matline 39 gs.pos (-142.892, 81.467,202.901)
    //qcerenkov::wavelength_sampled_bndtex idx   6197 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength  97.833 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6198 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 131.293 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6199 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 100.279 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6200 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 443.520 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6201 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength  86.645 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6202 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 360.446 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6203 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 314.119 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   3757 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  91.699 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3758 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 131.925 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3759 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  81.503 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3760 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 120.012 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3761 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 269.960 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3762 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  96.575 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3763 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 205.336 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3764 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 361.207 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3765 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 211.987 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3766 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 135.961 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3767 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  86.554 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3768 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  98.982 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3769 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 142.503 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3770 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  81.366 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3771 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 116.584 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3772 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 182.238 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3773 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 130.365 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    2026-05-08 16:05:44.483 INFO  [2439823] [QSim::simulate@736]  eventID     99 gs (138, 6, 4, ) ht (5, 4, 4, ) tot_dt   0.001790 server_settings HitCompOneName:hit,PhotonCompOneName:photon tree_digest 79f17049c1f5806abe058cf4449eb712 SGenstep::Brief (138, 6, 4, )
           DsG4Scintillation_r4695 : gs    134 ph       8988 idx 0/1 1/1 2/265 3/74 4/24 5/10 6/8 7/4 8/82 9/23 ... 
               G4Cerenkov_modified : gs      4 ph         27 idx 36/2 53/17 58/1 95/7 
     total_ph 9015



mono-check shows same CK issue : likely to be from recent SSim to stree rejig ?
---------------------------------------------------------------------------------

::

    mono-check 

    Begin of Event --> 99
    2026-05-08 16:22:29.338 [junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    //qcerenkov::wavelength_sampled_bndtex idx   4387 sampledRI   1.263 cosTheta   1.195 sin2Theta   0.000 wavelength 179.010 count 100 gsid 58 gs.matline 39 gs.pos (-134.874, 79.721,202.973)
    //qcerenkov::wavelength_sampled_bndtex idx   6197 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength  97.833 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6198 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 131.293 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6199 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 100.279 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6200 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 443.520 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6201 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength  86.645 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6202 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 360.446 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   6203 sampledRI   1.263 cosTheta   1.129 sin2Theta   0.000 wavelength 314.119 count 100 gsid 95 gs.matline 39 gs.pos ( -7.978, 29.221, 33.671)
    //qcerenkov::wavelength_sampled_bndtex idx   3757 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  91.699 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3758 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 131.925 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3759 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  81.503 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3760 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 120.012 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3761 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 269.960 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3762 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  96.575 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3763 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 205.336 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3764 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 361.207 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3765 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 211.987 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3766 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 135.961 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3767 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  86.554 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3768 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  98.982 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3769 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 142.503 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3770 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength  81.366 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3771 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 116.584 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3772 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 182.238 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   3773 sampledRI   1.263 cosTheta   1.093 sin2Theta   0.000 wavelength 130.365 count 100 gsid 53 gs.matline 39 gs.pos (-134.891, 79.689,202.711)
    //qcerenkov::wavelength_sampled_bndtex idx   1927 sampledRI   1.263 cosTheta   1.219 sin2Theta   0.000 wavelength 204.238 count 100 gsid 36 gs.matline 39 gs.pos (-142.892, 81.467,202.901)
    //qcerenkov::wavelength_sampled_bndtex idx   1928 sampledRI   1.263 cosTheta   1.219 sin2Theta   0.000 wavelength 382.924 count 100 gsid 36 gs.matline 39 gs.pos (-142.892, 81.467,202.901)
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    2026-05-08 16:22:29.342 ]junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 5 hitCollectionAlt -1 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2026-05-08 08:22:29.342537000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2026-05-08 08:22:29.342676000Z
    end of event action 





qcerenkov::wavelength_sampled_bndtex
--------------------------------------


::

    285 inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_bndtex(float& wavelength, float& cosTheta, float& sin2Theta, RNG& rng, const scerenkov& gs, unsigned long long idx, int gsid ) const
    286 {
    287     //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline );
    288     float u0 ;
    289     float u1 ;
    290     float w ;
    291     float sampledRI ;
    292     float u_maxSin2 ;
    293 
    294     unsigned count = 0 ;
    295 
    296     do {
    297         u0 = curand_uniform(&rng) ;
    298 
    299         w = gs.Wmin + u0*(gs.Wmax - gs.Wmin) ;
    300 
    301         wavelength = gs.Wmin*gs.Wmax/w ; // reciprocalization : arranges flat energy distribution, expressed in wavelength
    302 
    303         float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u);
    304 
    305         sampledRI = props.x ;
    306 
    307         //printf("//qcerenkov::wavelength_sampled_bndtex count %d wavelength %10.4f sampledRI %10.4f \n", count, wavelength, sampledRI );
    308 
    309         cosTheta = gs.BetaInverse / sampledRI ;
    310 
    311         sin2Theta = fmaxf( 0.f, (1.f - cosTheta)*(1.f + cosTheta));
    312 
    313         u1 = curand_uniform(&rng) ;
    314 
    315         u_maxSin2 = u1*gs.maxSin2 ;
    316 
    317         count += 1 ;
    318 
    319     } while ( u_maxSin2 > sin2Theta && count < 100 );
    320 
    321 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    322     if(count > 50)
    323     printf("//qcerenkov::wavelength_sampled_bndtex idx %6lld sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d gsid %d gs.matline %d gs.pos (%7.3f,%7.3f,%7.3f)\n",
    324               idx , sampledRI, cosTheta, sin2Theta, wavelength, count, gsid, gs.matline, gs.pos.x, gs.pos.y, gs.pos.z );
    325 
    326 #endif
    327 
    328 }



revive QCerenkovTest.sh
--------------------------


note no propcom.npy ?::


    [lo] A[blyth@localhost standard]$ l
    total 104220
        4 -rw-r--r--. 1 blyth blyth      116 May  8 16:16 icdf_meta.txt
        4 -rw-r--r--. 1 blyth blyth        3 May  8 16:16 icdf_names.txt
      100 -rw-r--r--. 1 blyth blyth    98432 May  8 16:16 icdf.npy
        4 -rw-r--r--. 1 blyth blyth       91 May  8 16:16 NPFold_index.txt
        0 -rw-r--r--. 1 blyth blyth        0 May  8 16:16 NPFold_names.txt
       32 -rw-r--r--. 1 blyth blyth    28928 May  8 16:16 optical.npy
        4 -rw-r--r--. 1 blyth blyth       61 May  8 16:16 bnd_meta.txt
       24 -rw-r--r--. 1 blyth blyth    23651 May  8 16:16 bnd_names.txt
    85616 -rw-r--r--. 1 blyth blyth 87667344 May  8 16:16 bnd.npy
       24 -rw-r--r--. 1 blyth blyth    23651 May  8 16:16 bd_names.txt
        8 -rw-r--r--. 1 blyth blyth     7328 May  8 16:16 bd.npy
       16 -rw-r--r--. 1 blyth blyth    13188 May  8 16:16 sur_names.txt
    17172 -rw-r--r--. 1 blyth blyth 17582272 May  8 16:16 sur.npy
        4 -rw-r--r--. 1 blyth blyth       35 May  8 16:16 mat_meta.txt
        4 -rw-r--r--. 1 blyth blyth      217 May  8 16:16 mat_names.txt
        8 -rw-r--r--. 1 blyth blyth     6216 May  8 16:16 energy.npy
     1144 -rw-r--r--. 1 blyth blyth  1169024 May  8 16:16 mat.npy
        4 -rw-r--r--. 1 blyth blyth       42 May  8 16:16 rayleigh_meta.txt
        4 -rw-r--r--. 1 blyth blyth      462 May  8 16:16 rayleigh_names.txt
       28 -rw-r--r--. 1 blyth blyth    27952 May  8 16:16 rayleigh.npy
        8 -rw-r--r--. 1 blyth blyth     6216 May  8 16:16 wavelength.npy
        4 drwxr-xr-x. 9 blyth blyth     4096 Apr  8 09:51 ..
        4 drwxr-xr-x. 2 blyth blyth     4096 Apr  8 09:51 .
    [lo] A[blyth@localhost standard]$ pwd
    /home/blyth/junosw/InstallArea/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/.opticks/GEOM/J26_1_1_opticks_Debug/CSGFoundry/SSim/stree/standard
    [lo] A[blyth@localhost standard]$ cd 



what happened to propcom ? its a red herring not used ?
----------------------------------------------------------

::

    [lo] A[blyth@localhost qudarap]$ opticks-f propcom
    ./ggeo/GGeo.cc:    const NP* propcom = SPropMockup::CombinationDemo();
    ./ggeo/GGeo.cc:    m_fold->add(snam::PROPCOM, propcom); 
    ./qudarap/QSim.cc:    const NP* propcom = ssim->get_propcom();
    ./qudarap/QSim.cc:    if( propcom )
    ./qudarap/QSim.cc:        QProp<float>* prop = new QProp<float>(propcom) ;
    ./qudarap/QSim.cc:        LOG(LEVEL) << "  propcom null, snam::PROPCOM " <<  snam::PROPCOM ;
    ./qudarap/tests/QCerenkovTest.cc:    const NP* propcom ;
    ./qudarap/tests/QCerenkovTest.cc:    propcom(ssim->get_propcom()),
    ./qudarap/tests/QCerenkovTest.cc:    prop(propcom ? new QProp<float>(propcom) : nullptr),
    ./qudarap/tests/QCerenkovTest.cc:    LOG(info) << " propcom " << ( propcom ? propcom->sstr() : "-" ) ;
    ./qudarap/tests/QCerenkovTest.cc:    NP_FATAL_ASSERT(propcom);
    ./qudarap/tests/QPropTest.cc:    const NP* propcom = SPropMockup::CombinationDemo();
    ./qudarap/tests/QPropTest.cc:    if(propcom == nullptr) std::cerr << "SPropMockup::CombinationDemo() giving null " << std::endl ; 
    ./qudarap/tests/QPropTest.cc:    if(propcom == nullptr) return 0 ; 
    ./qudarap/tests/QPropTest.cc:    std::cout << " propcom " << ( propcom ? propcom->sstr() : "-" ) << std::endl ; 
    ./qudarap/tests/QPropTest.cc:    QPropTest<float> qpt(propcom, 0.f, 16.f, nx ) ; 
    ./qudarap/tests/QPropTest.h:    QPropTest( const NP* propcom, T x0, T x1, int nx_ ); 
    ./qudarap/tests/QPropTest.h:inline QPropTest<T>::QPropTest( const NP* propcom, T x0, T x1, int nx_ )
    ./qudarap/tests/QPropTest.h:    qprop(new QProp<T>(propcom)),
    ./qudarap/tests/QProp_test.cc:    const NP* propcom = SPropMockup::CombinationDemo();
    ./qudarap/tests/QProp_test.cc:    std::cout << " propcom " << ( propcom ? propcom->sstr() : "-" ) << std::endl ; 
    ./qudarap/tests/QProp_test.cc:    QPropTest<float> qpt(propcom, 0.f, 16.f, nx ) ; 
    ./sysrap/SSim.cc:const NP* SSim::get_propcom() const {  return get(snam::PROPCOM);  }
    ./sysrap/SPropMockup.h:    const NP* propcom = Combination( DEMO_BASE, DEMO_RELP);
    ./sysrap/SPropMockup.h:    return propcom ;  
    ./sysrap/SSim.hh:    const NP* get_propcom() const ;
    ./sysrap/snam.h:    static constexpr const char* PROPCOM = "propcom.npy" ;
    [lo] A[blyth@localhost opticks]$ 


