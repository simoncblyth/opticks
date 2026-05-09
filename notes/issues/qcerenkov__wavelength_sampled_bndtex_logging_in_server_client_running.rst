FIXED qcerenkov__wavelength_sampled_bndtex_logging_in_server_client_running
==============================================================================

Overview : This issue reveals need for big QCerenkov cleanup
--------------------------------------------------------------

* issue is not server-client related : get same problem in mono-running

* DONE : QCerenkov split off old ICDF impl behind QCERENKOV_ICDF_OLD
* WIP : revive + rework QCerenkovTest.sh for current qcerenkov.h bnd based impl

  * actually QSimTest.sh more useful, used that to debug the issue


ISSUE FOUND BY ADDING ROUNDTRIP TEST TO QTex::uploadMeta
----------------------------------------------------------


::

    137 /**
    138 QTex:uploadMeta
    139 ------------------
    140 
    141 This is invoked by higher level users, not automatically by QTex,
    142 for example see QBnd::MakeBoundaryTex.
    143 
    144 Having correct tex metadata device side is vital. To ensure that
    145 this method does a roundtrip test downloading the metadata and
    146 comparing with expection.
    147 
    148 Formerly a "sizeof(quad)" vs "sizeof(quad4)" typo here
    149 caused incorrect flat qcerenkov wavelength distribution
    150 and kernel warnings, see::
    151 
    152   ~/o/notes/issues/qcerenkov__wavelength_sampled_bndtex_logging_in_server_client_running.rst
    153 
    154 **/
    155 
    156 template<typename T>
    157 void QTex<T>::uploadMeta()
    158 {
    159 #if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
    160     d_meta = meta ;
    161 #else
    162     size_t size = sizeof(*meta);
    163 
    164     d_meta = nullptr ;
    165     QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_meta ), size ));
    166 
    167     QUDA_CHECK( cudaMemcpy( d_meta         , meta   , size, cudaMemcpyHostToDevice ));
    168     QUDA_CHECK( cudaMemcpy( meta_roundtrip , d_meta , size, cudaMemcpyDeviceToHost ));
    169 
    170     bool roundtrip = 0 == memcmp(meta, meta_roundtrip, size );
    171     LOG_IF(fatal, !roundtrip )
    172         << " roundtrip " << ( roundtrip ? "YES" : "NO " ) << "\n"
    173         << " meta.desc " << meta->desc_tex_meta()
    174         << " meta_roundtrip.desc " << meta_roundtrip->desc_tex_meta()
    175         ;
    176 
    177     NP_FATAL_ASSERT(roundtrip);
    178 #endif
    179 }
    180 




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



QSimTest.sh reproduces issue
-----------------------------

::

    NUM=1 TEST=cerenkov_generate ~/o/qudarap/tests/QSimTest.sh




Something wrong with bnd prop access or the props are unexpectedly flat with wavelength::





    [lo] A[blyth@localhost qudarap]$ NUM=1 TEST=cerenkov_generate ~/o/qudarap/tests/QSimTest.sh 
    === ephoton.sh : TEST cerenkov_generate : unset environment : will use C++ defaults in quad4::ephoton for p0
    2026-05-09 10:05:03.554 INFO  [3090250] [main@792] [ TEST cerenkov_generate
    2026-05-09 10:05:03.582 INFO  [3090250] [QSimTest::EventConfig@538] [ cerenkov_generate
    2026-05-09 10:05:03.582 INFO  [3090250] [QSimTest::EventConfig@557] ] cerenkov_generate
    2026-05-09 10:05:03.582 INFO  [3090250] [main@811] [SSim::Load
    2026-05-09 10:05:03.606 INFO  [3090250] [SEventConfig::SetDevice@1893] SEventConfig::DescDevice
    ...


    SPMT::init_pmtNum
    2026-05-09 10:05:05.205 INFO  [3090250] [QSimTest::main@704]  num 1 type 9 subfold cerenkov_generate ni_tranche_size 100000 print_id -1
    //QSim_dbg_gs_generate sim 0x7fc514c4ca00 dbg 0x7fc514c07a00 photon 0x7fc514c4cc00 num_photon 1 type 9 name cerenkov_generate 
    //qcerenkov::wavelength_sampled_bndtex count 0 wavelength   174.2159 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 1 wavelength   104.9540 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 2 wavelength    82.0394 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 3 wavelength   110.4011 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 4 wavelength   680.7383 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 5 wavelength   352.6079 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 6 wavelength    98.9200 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 7 wavelength   124.1754 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 8 wavelength    84.9976 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 9 wavelength    84.3848 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 10 wavelength   158.3036 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 11 wavelength   113.7430 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 12 wavelength   107.7665 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 13 wavelength    94.4758 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 14 wavelength   115.4113 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 15 wavelength   106.9285 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 16 wavelength   246.7202 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 17 wavelength    84.4979 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 18 wavelength   126.8585 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 19 wavelength    83.9819 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 20 wavelength    93.2821 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 21 wavelength   317.7622 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 22 wavelength   120.1270 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 23 wavelength    85.7521 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 24 wavelength   117.4440 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 25 wavelength   354.6295 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 26 wavelength   190.6907 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 27 wavelength   172.5606 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 28 wavelength   618.8285 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 29 wavelength   141.1729 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 30 wavelength   118.9044 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 31 wavelength   108.0496 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 32 wavelength   127.9489 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 33 wavelength    84.9957 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 34 wavelength   167.7991 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qcerenkov::wavelength_sampled_bndtex count 35 wavelength   153.8783 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 





Smoking gun : problem with bnd metadata describing the texture
-----------------------------------------------------------------

::

    SPMT::init_pmtNum
    2026-05-09 10:13:16.803 INFO  [3091792] [QSimTest::main@704]  num 1 type 9 subfold cerenkov_generate ni_tranche_size 100000 print_id -1
    //QSim_dbg_gs_generate sim 0x7fd228c4ca00 dbg 0x7fd228c07a00 photon 0x7fd228c4cc00 num_photon 1 type 9 name cerenkov_generate 
    //qbnd.boundary_lookup nm   174.2159 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 0 wavelength   174.2159 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   104.9540 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 1 wavelength   104.9540 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    82.0394 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 2 wavelength    82.0394 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   110.4011 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 3 wavelength   110.4011 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   680.7383 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 4 wavelength   680.7383 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   352.6079 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 5 wavelength   352.6079 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    98.9200 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 6 wavelength    98.9200 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   124.1754 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 7 wavelength   124.1754 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    84.9976 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  
    //qcerenkov::wavelength_sampled_bndtex count 8 wavelength    84.9976 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    84.3848 nm0     0.0000 nms     0.0000  x        inf nx 761 ny 3600 y     0.0218 props (     1.2632  3270.3213 160987.5000     0.2000 )  

bmeta.q1.f all zero
-----------------------

::

    2026-05-09 10:23:51.985 INFO  [3093718] [QSimTest::main@704]  num 1 type 9 subfold cerenkov_generate ni_tranche_size 100000 print_id -1
    //QSim_dbg_gs_generate sim 0x7ff2fac4ca00 dbg 0x7ff2fac07a00 photon 0x7ff2fac4cc00 num_photon 1 type 9 name cerenkov_generate 
    //qbnd.boundary_lookup nm   174.2159 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 0 wavelength   174.2159 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   104.9540 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 1 wavelength   104.9540 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    82.0394 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 2 wavelength    82.0394 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   110.4011 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 3 wavelength   110.4011 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   680.7383 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 4 wavelength   680.7383 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   352.6079 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 5 wavelength   352.6079 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    98.9200 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 6 wavelength    98.9200 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm   124.1754 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 
    //qcerenkov::wavelength_sampled_bndtex count 7 wavelength   124.1754 sampledRI     1.2632 props (    1.2632, 3270.3213,160987.5000,    0.2000) 
    //qbnd.boundary_lookup nm    84.9976 matline_Water 171 matline_LS 39  bmeta.q1.f (     0.000,     0.000,     0.000,     0.000) 



qbnd.h
-------

::

     30 struct qbnd
     31 {
     32     cudaTextureObject_t boundary_tex ;
     33     quad4*              boundary_meta ;
     34     unsigned            boundary_tex_MaterialLine_Water ;
     35     unsigned            boundary_tex_MaterialLine_LS ;
     36     quad*               optical ;
     37 



boundary_meta from QTex::d_meta
--------------------------------

::

     46 qbnd* QBnd::MakeInstance(const QTex<float4>* tex, const std::vector<std::string>& names )
     47 {
     48     qbnd* qb = new qbnd ;
     49 
     50     qb->boundary_tex = tex->texObj ;
     51     qb->boundary_meta = tex->d_meta ;
     52     qb->boundary_tex_MaterialLine_Water = SBnd::GetMaterialLine("Water", names) ;
     53     qb->boundary_tex_MaterialLine_LS    = SBnd::GetMaterialLine("LS", names) ;
     54 
     55     const QOptical* optical = QOptical::Get() ;
     56     //assert( optical ); 
     57 
     58 #if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
     59 #else
     60     LOG(LEVEL) << " optical " << ( optical ? optical->desc() : "MISSING" ) ;
     61 #endif
     62 
     63     qb->optical = optical ? optical->d_optical : nullptr ;
     64 
     65     assert( qb->optical != nullptr );
     66     assert( qb->boundary_meta != nullptr );
     67     return qb ;
     68 }
     69 
     70 
     71 /**
     72 QBnd::QBnd
     73 ------------
     74 
     75 Narrows the NP array if wide and creates GPU texture 
     76 
     77 **/
     78 
     79 QBnd::QBnd(const NP* buf)
     80     :
     81     dsrc(buf->ebyte == 8 ? buf : nullptr),
     82     src(NP::MakeNarrowIfWide(buf)),
     83     sbn(new SBnd(src)),
     84     tex(MakeBoundaryTex(src)),
     85     qb(MakeInstance(tex, buf->names)),
     86     d_qb(nullptr)
     87 {
     88     init();
     89 }



::

    109 template<typename T>
    110 void QTex<T>::setMetaDomainX( const quad* domx )
    111 {
    112     meta->q1.f.x = domx->f.x ;
    113     meta->q1.f.y = domx->f.y ;
    114     meta->q1.f.z = domx->f.z ;
    115     meta->q1.f.w = domx->f.w ;
    116 }
    117 
    118 template<typename T>
    119 void QTex<T>::setMetaDomainY( const quad* domy )
    120 {
    121     meta->q2.f.x = domy->f.x ;
    122     meta->q2.f.y = domy->f.y ;
    123     meta->q2.f.z = domy->f.z ;
    124     meta->q2.f.w = domy->f.w ;
    125 }
    126 



domain metadata is being set
-----------------------------


::

    BP=QBnd::MakeBoundaryTex NUM=1  TEST=cerenkov_generate ~/o/qudarap/tests/QSimTest.sh


    (gdb) c
    Continuing.

    Thread 1 "QSimTest" hit Breakpoint 3, QBnd::MakeBoundaryTex (buf=0x14b80ec0) at /home/blyth/opticks/qudarap/QBnd.cc:192
    192	    return btex ; 
    (gdb) p domainX.f
    $2 = {x = 60, y = 820, z = 1, w = 760}
    (gdb) 



HUH roundtrip mismatch ?
--------------------------

::

    2026-05-09 11:12:07.068 INFO  [3103101] [QTex<T>::uploadMeta@160]  meta.desc quad4::desc_tex_meta
     q0.i (   761,  3600,     0,     0) 
     q1.f (60.000,820.000, 1.000,760.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 

    2026-05-09 11:12:07.068 INFO  [3103101] [QTex<T>::uploadMeta@161]  meta_roundtrip.desc quad4::desc_tex_meta
     q0.i (   761,  3600,     0,     0) 
     q1.f ( 0.000, 0.000, 0.000, 0.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 

    2026-05-09 11:12:07.072 INFO  [3103101] [QTex<T>::uploadMeta@160]  meta.desc quad4::desc_tex_meta
     q0.i (  4096,     3,     0,    20) 
     q1.f ( 0.000, 0.000, 0.000, 0.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 

    2026-05-09 11:12:07.072 INFO  [3103101] [QTex<T>::uploadMeta@161]  meta_roundtrip.desc quad4::desc_tex_meta
     q0.i (  4096,     3,     0,    20) 
     q1.f ( 0.000, 0.000, 0.000, 0.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 



BINGO : it was sizeof(quad) vs sizeof(quad4) typo !
------------------------------------------------------

Add roundtrip test::

    2026-05-09 11:31:18.883 INFO  [3105935] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000000 SEventConfig::MaxCurand 1000000000000
    2026-05-09 11:31:18.952 FATAL [3105935] [QTex<T>::uploadMeta@161]  roundtrip NO 
     meta.desc quad4::desc_tex_meta
     q0.i (   761,  3600,     0,     0) 
     q1.f (60.000,820.000, 1.000,760.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 
     meta_roundtrip.desc quad4::desc_tex_meta
     q0.i (   761,  3600,     0,     0) 
     q1.f ( 0.000, 0.000, 0.000, 0.000) 
     q2.f ( 0.000, 0.000, 0.000, 0.000) 
     q3.f ( 0.000, 0.000, 0.000, 0.000) 

    NP_FATAL_ASSERT 'roundtrip' failed at /home/blyth/opticks/qudarap/QTex.cc:167



