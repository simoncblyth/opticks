cxs_min_rare_CUDA_exception
=============================



Some error for photon between 7M and 8M::

    P[blyth@localhost opticks]$ TEST=ref7 CSGOptiX/cxs_min.sh
    ...
    2024-10-14 18:50:27.749  749764765 : [CSGOptiX/cxs_min.sh 
    2024-10-14 18:50:43.052  052718151 : ]CSGOptiX/cxs_min.sh 
    ... 



    P[blyth@localhost opticks]$ TEST=ref8 CSGOptiX/cxs_min.sh
    ...
    2024-10-14 18:53:37.838  838733666 : [CSGOptiX/cxs_min.sh 
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1077)

    CSGOptiX/cxs_min.sh: line 365: 80441 Aborted                 (core dumped) $bin
    CSGOptiX/cxs_min.sh run error



Binary search for the problem photon::

    P[blyth@localhost opticks]$ TEST=refX X=7000000 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7125000 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7140625 CSGOptiX/cxs_min.sh #  ok  
    P[blyth@localhost opticks]$ TEST=refX X=7148437 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7150390 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7151366 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7151854 CSGOptiX/cxs_min.sh #  ok 
    P[blyth@localhost opticks]$ TEST=refX X=7151915 CSGOptiX/cxs_min.sh #  ok 
    P[blyth@localhost opticks]$ TEST=refX X=7151945 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7151960 CSGOptiX/cxs_min.sh #  ok
    P[blyth@localhost opticks]$ TEST=refX X=7151968 CSGOptiX/cxs_min.sh #  ok 
    P[blyth@localhost opticks]$ TEST=refX X=7151969 CSGOptiX/cxs_min.sh #  ok 

    P[blyth@localhost opticks]$ TEST=refX X=7151970 CSGOptiX/cxs_min.sh # NOK   <<<<< FIRST BAD SLOT 
    P[blyth@localhost opticks]$ TEST=refX X=7151972 CSGOptiX/cxs_min.sh # NOK 
    P[blyth@localhost opticks]$ TEST=refX X=7151976 CSGOptiX/cxs_min.sh # NOK 
    P[blyth@localhost opticks]$ TEST=refX X=7152098 CSGOptiX/cxs_min.sh # NOK 
    P[blyth@localhost opticks]$ TEST=refX X=7152343 CSGOptiX/cxs_min.sh # NOK
    P[blyth@localhost opticks]$ TEST=refX X=7156250 CSGOptiX/cxs_min.sh # NOK  
    P[blyth@localhost opticks]$ TEST=refX X=7187500 CSGOptiX/cxs_min.sh # NOK
    P[blyth@localhost opticks]$ TEST=refX X=7250000 CSGOptiX/cxs_min.sh # NOK 
    P[blyth@localhost opticks]$ TEST=refX X=7500000 CSGOptiX/cxs_min.sh # NOK



Try to dump the problem slot::

    P[blyth@localhost opticks]$ TEST=refX X=7151970 PIDX=7151969 CSGOptiX/cxs_min.sh  


    //qsim.propagate_to_boundary.tail.SAIL idx 7151969 : post = np.array([15953.52051,-12802.56738,2317.32715, 118.93829]) ;  sail_time_delta =    0.03774   
    //qsim.propagate.body idx 7151969 bounce 10 command 3 flag 0 s.optical.x 0 s.optical.y 1 
    //qsim.propagate.body.WITH_CUSTOM4 idx 7151969  BOUNDARY ems 1 lposcost   0.399 
    //qsim.propagate_at_boundary.head idx 7151969 : theTransmittance = -1.00000000 
    //qsim.propagate_at_boundary.head idx 7151969 : nrm = np.array([0.98010135,0.19088598,0.05444238]) ; lnrm = 1.00000000  
    //qsim.propagate_at_boundary.head idx 7151969 : pos = np.array([15953.52051,-12802.56738,2317.32715]) ; lpos = 20586.17382812 
    //qsim.propagate_at_boundary.head idx 7151969 : mom0 = np.array([-0.18258519,-0.48212054,0.85686779]) ; lmom0 = 0.99999994 
    //qsim.propagate_at_boundary.head idx 7151969 : pol0 = np.array([0.16947818,-0.87390584,-0.45559391]) ; lpol0 = 1.00000000 
    //qsim.propagate_at_boundary.head idx 7151969 : n1,n2,eta = (1.35398555,1.48426318,0.91222739) 
    //qsim.propagate_at_boundary.head idx 7151969 : c1 = 0.22433214 ; normal_incidence = 0 
    //qsim.propagate_at_boundary.body idx 7151969 : TransCoeff = 0.85388458 ; n1c1 = 0.30374247 ; n2c2 = 0.67972046 
    //qsim.propagate_at_boundary.body idx 7151969 : E2_t = np.array([-0.61748815,0.01668877]) ; lE2_t = 0.61771363 
    //qsim.propagate_at_boundary.body idx 7151969 : A_trans = np.array([-0.19477613,0.87198204,0.44912088]) ; lA_trans = 1.00000000 
    //qsim.propagate_at_boundary.body idx 7151969 : u_reflect     0.1863 TransCoeff     0.8539 reflect 0 
    //qsim.propagate_at_boundary.body idx 7151969 : mom0 = np.array([-0.18258519,-0.48212054,0.85686779]) ; lmom0 = 0.99999994 
    //qsim.propagate_at_boundary.body idx 7151969 : pos = np.array([15953.52051,-12802.56738,2317.32715]) ; lpos = 20586.17382812 
    //qsim.propagate_at_boundary.body idx 7151969 : nrm = np.array([0.98010135,0.19088598,0.05444238]) ; lnrm = 1.00000000 
    //qsim.propagate_at_boundary.body idx 7151969 : n1 = 1.35398555 ; n2 = 1.48426318 ; eta = 0.91222739  
    //qsim.propagate_at_boundary.body idx 7151969 : c1 = 0.22433214 ; eta_c1 = 0.20464192 ; c2 = 0.45795146 ; eta_c1__c2 = -0.25330955 
    //qsim.propagate_at_boundary.tail idx 7151969 : reflect 0 tir 0 TransCoeff     0.8539 u_reflect     0.1863 
    //qsim.propagate_at_boundary.tail idx 7151969 : mom1 = np.array([-0.41482824,-0.48815683,0.76786751]) ; lmom1 = 1.00000000  
    //qsim.propagate_at_boundary.tail idx 7151969 : pol1 = np.array([0.17069212,-0.87067097,-0.46129841]) ; lpol1 = 1.00000000 
    //qsim.propagate.tail idx 7151969 bounce 10 command 2 flag 2048 ctx.s.optical.y(ems) 1 

    //qsim.propagate.head idx 7151969 : bnc 11 cosTheta -0.42115211 
    //qsim.propagate.head idx 7151969 : mom = np.array([-0.41482824,-0.48815683,0.76786751]) ; lmom = 1.00000000  
    //qsim.propagate.head idx 7151969 : pos = np.array([15953.52051,-12802.56738,2317.32715]) ; lpos = 20586.17382812 
    //qsim.propagate.head idx 7151969 : nrm = np.array([(0.98242778,0.16502818,0.08718470]) ; lnrm = 0.99999988  
    //qsim.propagate_to_boundary.head idx 7151969 : u_absorption 0.37148836 logf(u_absorption) -0.99023771 absorption_length  1562.9586 absorption_distance 1547.700562 
    //qsim.propagate_to_boundary.head idx 7151969 : post = np.array([15953.52051,-12802.56738,2317.32715, 118.93829]) 
    //qsim.propagate_to_boundary.head idx 7151969 : distance_to_boundary    11.2398 absorption_distance  1547.7006 scattering_distance 1075134005248.0000 
    //qsim.propagate_to_boundary.head idx 7151969 : u_scattering     0.3413 u_absorption     0.3715 
    //qsim.propagate_to_boundary.tail.SAIL idx 7151969 : post = np.array([15948.85840,-12808.05371,2325.95776, 118.99557]) ;  sail_time_delta =    0.05728   
    //qsim.propagate.body idx 7151969 bounce 11 command 3 flag 0 s.optical.x 36 s.optical.y 4 
    //qsim.propagate.body.WITH_CUSTOM4 idx 7151969  BOUNDARY ems 4 lposcost   0.414 
    //qsim::propagate_at_surface_CustomART idx 7151969 : mom = np.array([-0.41482824,-0.48815683,0.76786751]) ; lmom = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx 7151969 : pol = np.array([0.17069212,-0.87067097,-0.46129841]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx 7151969 : nrm = np.array([0.98242778,0.16502818,0.08718470]) ; lnrm = 0.99999988 
    //qsim::propagate_at_surface_CustomART idx 7151969 : cross_mom_nrm = np.array([-0.16927959,0.79054105,0.41112047]) ; lcross_mom_nrm = 0.90698999  
    //qsim::propagate_at_surface_CustomART idx 7151969 : dot_pol_cross_mom_nrm = -0.90684503 
    //qsim::propagate_at_surface_CustomART idx 7151969 : minus_cos_theta = -0.42115211 
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1077)

    CSGOptiX/cxs_min.sh: line 372: 147704 Aborted                 (core dumped) $bin
    CSGOptiX/cxs_min.sh run error
    P[blyth@localhost opticks]$ 


Probably the qpmt::get_lpmtid_ARTE call ? Add pre-ARTE dumping to confirm::

    1754        ctx.idx, normal->x, normal->y, normal->z, length(*normal) );
    1755     printf("//qsim::propagate_at_surface_CustomART idx %7d : cross_mom_nrm = np.array([%10.8f,%10.8f,%10.8f]) ; lcross_mom_nrm = %10.8f  \n",
    1756            ctx.idx, cross_mom_nrm.x, cross_mom_nrm.y, cross_mom_nrm.z, length(cross_mom_nrm)  );
    1757     printf("//qsim::propagate_at_surface_CustomART idx %7d : dot_pol_cross_mom_nrm = %10.8f \n", ctx.idx, dot_pol_cross_mom_nrm );
    1758     printf("//qsim::propagate_at_surface_CustomART idx %7d : minus_cos_theta = %10.8f \n", ctx.idx, minus_cos_theta );
    1759     }
    1760 #endif
    1761 
    1762     if(lpmtid < 0 )
    1763     {
    1764         flag = NAN_ABORT ;
    1765 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1766         //if( ctx.idx == base->pidx ) 
    1767         printf("//qsim::propagate_at_surface_CustomART idx %7d lpmtid %d : ERROR NOT-A-SENSOR : NAN_ABORT \n", ctx.idx, lpmtid );
    1768 #endif
    1769         return BREAK ;
    1770     }
    1771 
    1772 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1773     if( ctx.idx == base->pidx )
    1774     printf("//qsim::propagate_at_surface_CustomART idx %d lpmtid %d wl %7.3f mct %7.3f dpcmn %7.3f pre-ARTE \n",
    1775            ctx.idx, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm );
    1776 #endif
    1777 
    1778     float ARTE[4] ;
    1779     if(lpmtid > -1) pmt->get_lpmtid_ARTE(ARTE, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm );
    1780 
    1781 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1782     if( ctx.idx == base->pidx )
    1783     printf("//qsim::propagate_at_surface_CustomART idx %d lpmtid %d wl %7.3f mct %7.3f dpcmn %7.3f ARTE ( %7.3f %7.3f %7.3f %7.3f ) \n",
    1784            ctx.idx, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm, ARTE[0], ARTE[1], ARTE[2], ARTE[3] );
    1785 #endif
    1786 
    1787 
    1788     const float& theAbsorption = ARTE[0];
    1789     //const float& theReflectivity = ARTE[1]; 
    1790     const float& theTransmittance = ARTE[2];
    1791     const float& theEfficiency = ARTE[3];
    1792 
    1793     float u_theAbsorption = curand_uniform(&rng);
    1794     int action = u_theAbsorption < theAbsorption  ? BREAK : CONTINUE ;
    1795 


lpmtid 50937 looks suspicious::

    //qsim.propagate_to_boundary.head idx 7151969 : u_scattering     0.3413 u_absorption     0.3715 
    //qsim.propagate_to_boundary.tail.SAIL idx 7151969 : post = np.array([15948.85840,-12808.05371,2325.95776, 118.99557]) ;  sail_time_delta =    0.05728   
    //qsim.propagate.body idx 7151969 bounce 11 command 3 flag 0 s.optical.x 36 s.optical.y 4 
    //qsim.propagate.body.WITH_CUSTOM4 idx 7151969  BOUNDARY ems 4 lposcost   0.414 
    //qsim::propagate_at_surface_CustomART idx 7151969 : mom = np.array([-0.41482824,-0.48815683,0.76786751]) ; lmom = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx 7151969 : pol = np.array([0.17069212,-0.87067097,-0.46129841]) ; lpol = 1.00000000 
    //qsim::propagate_at_surface_CustomART idx 7151969 : nrm = np.array([0.98242778,0.16502818,0.08718470]) ; lnrm = 0.99999988 
    //qsim::propagate_at_surface_CustomART idx 7151969 : cross_mom_nrm = np.array([-0.16927959,0.79054105,0.41112047]) ; lcross_mom_nrm = 0.90698999  
    //qsim::propagate_at_surface_CustomART idx 7151969 : dot_pol_cross_mom_nrm = -0.90684503 
    //qsim::propagate_at_surface_CustomART idx 7151969 : minus_cos_theta = -0.42115211 
    //qsim::propagate_at_surface_CustomART idx 7151969 lpmtid 50937 wl 420.000 mct  -0.421 dpcmn  -0.907 pre-ARTE 
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1077)

    CSGOptiX/cxs_min.sh: line 372: 169951 Aborted                 (core dumped) $bin
    CSGOptiX/cxs_min.sh run error
    P[blyth@localhost opticks]$ 


::

    304 template<typename F>
    305 inline QPMT_METHOD void qpmt<F>::get_lpmtid_ARTE(
    306     F* arte4,
    307     int lpmtid,
    308     F wavelength_nm,
    309     F minus_cos_theta,
    310     F dot_pol_cross_mom_nrm ) const
    311 {
    312     const F energy_eV = hc_eVnm/wavelength_nm ;
    313 
    314     F spec[16] ;
    315     get_lpmtid_stackspec( spec, lpmtid, energy_eV );
    316 
    317     const F* ss = spec ;
    318     const F& _qe = spec[15] ;
    319 
    320 #ifdef MOCK_CURAND_DEBUG
    321     printf("//qpmt::get_lpmtid_ARTE lpmtid %d energy_eV %7.3f _qe %7.3f \n", lpmtid, energy_eV, _qe );
    322 #endif
    323 
    324 

    156 template<typename F>
    157 inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec( F* spec, int lpmtid, F energy_eV ) const
    158 {
    159     const int& lpmtcat = i_lcqs[lpmtid*2+0] ;
    160     // printf("//qpmt::get_lpmtid_stackspec lpmtid %d lpmtcat %d \n", lpmtid, lpmtcat );  
    161 
    162     const F& qe_scale = lcqs[lpmtid*2+1] ;
    163     const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ;
    164     const F qe = qe_scale*qe_shape ;
    165 
    166     spec[0*4+3] = lpmtcat ;
    167     spec[1*4+3] = qe_scale ;
    168     spec[2*4+3] = qe_shape ;
    169     spec[3*4+3] = qe ;
    170 
    171     get_lpmtcat_stackspec( spec, lpmtcat, energy_eV );
    172 }


::

    102 template<typename T>
    103 inline QPMT<T>::QPMT(const NPFold* jpmt )
    104     :
    105     ExecutableName(sproc::ExecutableName()),
    106     src_rindex(   jpmt->get("rindex")),
    107     src_thickness(jpmt->get("thickness")),
    108     src_qeshape(  jpmt->get("qeshape")),
    109     src_lcqs(     jpmt->get_optional("lcqs")),
    110     rindex3(  NP::MakeCopy3D(src_rindex)),   // make copy and change shape to 3D
    111     rindex(   NP::MakeWithType<T>(rindex3)), // adopt template type, potentially narrowing
    112     rindex_prop(new QProp<T>(rindex)),
    113     qeshape(   NP::MakeWithType<T>(src_qeshape)), // adopt template type, potentially narrowing
    114     qeshape_prop(new QProp<T>(qeshape)),
    115     thickness(NP::MakeWithType<T>(src_thickness)),
    116     lcqs(src_lcqs ? NP::MakeWithType<T>(src_lcqs) : nullptr),
    117     i_lcqs( lcqs ? (int*)lcqs->cvalues<T>() : nullptr ),    // CPU side lookup lpmtid->lpmtcat 0/1/2
    118     pmt(new qpmt<T>()),                    // host-side qpmt.h instance 
    119     d_pmt(nullptr)                         // device-side pointer set at upload in init
    120 {
    121     init();
    122 }
    123 


QSim.cc::

    0180     const NPFold* spmt_f = ssim->get_spmt_f() ;
     181     QPMT<float>* qpmt = spmt_f ? new QPMT<float>(spmt_f) : nullptr ;
     182     LOG_IF(LEVEL, qpmt == nullptr )
     183         << " NO QPMT instance "
     184         << " spmt_f " << ( spmt_f ? "YES" : "NO " )
     185         << " qpmt " << ( qpmt ? "YES" : "NO " )
     186         ;
     187 
     188     LOG(LEVEL)
     189         << QPMT<float>::Desc()
     190         << std::endl
     191         << " spmt_f " << ( spmt_f ? "YES" : "NO " )
     192         << " qpmt " << ( qpmt ? "YES" : "NO " )
     193         ;


::


    055     static constexpr const char* JPMT_RELP = "extra/jpmt" ;


    310 const NPFold* SSim::get_jpmt() const
    311 {
    312     const NPFold* f = top ? top->find_subfold(JPMT_RELP) : nullptr ;
    313     return f ;
    314 }
    315 const SPMT* SSim::get_spmt() const
    316 {
    317     const NPFold* jpmt = get_jpmt();
    318     return jpmt ? new SPMT(jpmt) : nullptr ;
    319 }
    320 const NPFold* SSim::get_spmt_f() const
    321 {
    322     const SPMT* spmt = get_spmt() ;
    323     const NPFold* spmt_f = spmt ? spmt->serialize() : nullptr ;
    324     return spmt_f ;
    325 }


Looks like rare issue from lpmtid being >= 17612::

    0465 inline void SPMT::init_lcqs()
     466 {
     467     assert( PMTSimParamData );
     468     const NP* lpmtCat = PMTSimParamData->get("lpmtCat") ;
     469     assert( lpmtCat && lpmtCat->uifc == 'i' && lpmtCat->ebyte == 4 );
     470     assert( lpmtCat->shape[0] == NUM_LPMT );
     471     const int* lpmtCat_v = lpmtCat->cvalues<int>();
     472 
     473     const NP* qeScale = PMTSimParamData->get("qeScale") ;
     474     assert( qeScale && qeScale->uifc == 'f' && qeScale->ebyte == 8 );
     475     assert( qeScale->shape[0] >= NUM_LPMT );  // SPMT, WPMT info after LPMT 
     476     const double* qeScale_v = qeScale->cvalues<double>();
     477 
     478     for(int i=0 ; i < NUM_LPMT ; i++ )
     479     {
     480         v_lcqs[i] = { TranslateCat(lpmtCat_v[i]), float(qeScale_v[i]) } ;
     481     }
     482     lcqs = NPX::ArrayFromVec<int,LCQS>( v_lcqs ) ;
     483 
     484     if(VERBOSE) std::cout
     485        << "SPMT::init_lcqs" << std::endl
     486        << " NUM_LPMT " << NUM_LPMT << std::endl
     487        << " lpmtCat " << ( lpmtCat ? lpmtCat->sstr() : "-" ) << std::endl
     488        << " qeScale " << ( qeScale ? qeScale->sstr() : "-" ) << std::endl
     489        << " lcqs " << ( lcqs ? lcqs->sstr() : "-" ) << std::endl
     490        ;
     491 
     492     assert( lcqs->shape[0] == NUM_LPMT );
     493     assert( NUM_LPMT == 17612 );
     494 }




Avoid the dud lpmtid::

    1762     if(lpmtid < 0 || lpmtid >= 17612 )
    1763     {
    1764         flag = NAN_ABORT ;
    1765 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1766         //if( ctx.idx == base->pidx ) 
    1767         printf("//qsim::propagate_at_surface_CustomART idx %7d lpmtid %d : ERROR NOT-A-SENSOR : NAN_ABORT \n", ctx.idx, lpmtid );
    1768 #endif
    1769         return BREAK ;
    1770     }
    1771 
    1772 #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    1773     if( ctx.idx == base->pidx )
    1774     printf("//qsim::propagate_at_surface_CustomART idx %d lpmtid %d wl %7.3f mct %7.3f dpcmn %7.3f pre-ARTE \n",
    1775            ctx.idx, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm );
    1776 #endif




That shows are getting dud lpmtid 3 times from 10M photons::

    P[blyth@localhost opticks]$ TEST=ref10 CSGOptiX/cxs_min.sh
    CSGOptiX/cxs_min.sh : FOUND A_CFBaseFromGEOM /cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/J_2024aug27 containing CSGFoundry/prim.npy
    ...  
      opticks_num_photon : M10 
      OPTICKS_NUM_PHOTON : M10 
    2024-10-14 20:18:09.754  754709454 : [CSGOptiX/cxs_min.sh 
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 7151969 lpmtid 50937 : ERROR NOT-A-SENSOR : NAN_ABORT 
    //qsim::propagate_at_surface_CustomART idx 8759857 lpmtid 51162 : ERROR NOT-A-SENSOR : NAN_ABORT 
    2024-10-14 20:18:30.389  389686877 : ]CSGOptiX/cxs_min.sh 




