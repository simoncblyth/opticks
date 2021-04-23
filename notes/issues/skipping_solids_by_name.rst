skipping_solids_by_name
===========================



Feature
---------

Want to skip the virtual "hat boxes" surrounding PMTs for the GPU geometry.
But do not want to use raw lvidx indices as they go stale too fast with 
geometry updates. So need to get from mesh/solid name to idx at OpticksDbg level.
Normally geometry info is not available down there.


Approach
-----------

1. introduce sysrap/SGeo pure virtual base protocol for ggeo/GGeo to inherit from 
2. arrange for Opticks to hold onto SGeo instance set with `Opticks::setGeo(const SGeo*)`
   and add `OpticksDbg::postgeometry` to act (translating names into indices) when 
   the geometry becomes available.

3. invoke that setter from `GGeo::deferred` after loading or creating geometry::

     594 void GGeo::deferred()
     595 {
     596     m_ok->setGeo((SGeo*)this);   //  for access to limited geometry info from lower levels 
     597     deferredCreateGParts();
     598 }

4. provide accessor via Opticks::

     758 bool Opticks::isSkipSolidIdx(unsigned lvIdx) const  // --skipsolidname
     759 {
     760     return m_dbg->isSkipSolidIdx(lvIdx);
     761 }
     
5. use that from `GParts::Create`::


     220 GParts* GParts::Create(const Opticks* ok, const GPts* pts, const std::vector<const NCSG*>& solids, unsigned& num_mismatch_pt, std::vector<glm::mat4>* mismatch_placements ) // static
     221 {
     222     LOG(LEVEL) << "[  deferred creation from GPts" ; 
     223 
     224     GParts* com = new GParts() ;
     225 
     226     unsigned num_pt = pts->getNumPt();
     227 
     228     LOG(LEVEL) << " num_pt " << num_pt ;
     229 
     230     unsigned verbosity = 0 ;
     231 
     232     std::vector<unsigned> mismatch_pt ;
     233 
     234     for(unsigned i=0 ; i < num_pt ; i++)
     235     {
     236         const GPt* pt = pts->getPt(i);
     237         int   lvIdx = pt->lvIdx ;
     238 
     239         bool deferredcsgskiplv = ok->isDeferredCSGSkipLV(lvIdx); //  --deferredcsgskiplv 
     240         bool skipsolidname     = ok->isSkipSolidIdx(lvIdx);      //   --skipsolidname 
     241         
     242         if(deferredcsgskiplv || skipsolidname)
     243         {
     244             LOG(info)
     245                 << " SKIPPING SOLID FROM ANALYTIC GEOMETRY VIA COMMANDLINE OPTION "
     246                 << " i " << i 
     247                 << " num_pt " << num_pt
     248                 << " lvIdx " << lvIdx 
     249                 << " deferredcsgskiplv " << deferredcsgskiplv
     250                 << " skipsolidname " << skipsolidname
     251                 ; 
     252             continue ;
     253         }




Testing 
----------

::

    epsilon:opticks blyth$ ggeo.sh 1:5/ --names 
    python3 /Users/blyth/opticks/ana/ggeo.py 1:5/ --names
    nrpo( 176632     1     0     0 )                        PMT_3inch_log_phys0x4437d00                             PMT_3inch_log0x4436df0  114 PMT_3inch_pmt_solid0x4436210 
    nrpo( 176633     1     0     1 )                       PMT_3inch_body_phys0x4437230                        PMT_3inch_body_log0x4436ce0  112 PMT_3inch_body_solid_ell_ell_helper0x44364d0 
    nrpo( 176634     1     0     2 )                     PMT_3inch_inner1_phys0x44372b0                      PMT_3inch_inner1_log0x4436f00  110 PMT_3inch_inner1_solid_ell_helper0x4436560 
    nrpo( 176635     1     0     3 )                     PMT_3inch_inner2_phys0x4437360                      PMT_3inch_inner2_log0x4437010  111 PMT_3inch_inner2_solid_ell_helper0x4436640 
    nrpo( 176636     1     0     4 )                       PMT_3inch_cntr_phys0x4437410                        PMT_3inch_cntr_log0x4437120  113 PMT_3inch_cntr_solid0x44366d0 
    nrpo(  70960     2     0     0 )                         pLPMT_NNVT_MCPPMT0x3cbba60                    NNVTMCPPMTlMaskVirtual0x3cb41a0  103 NNVTMCPPMTsMask_virtual0x3cb3b40 
    nrpo(  70961     2     0     1 )                           NNVTMCPPMTpMask0x3c9fe00                           NNVTMCPPMTlMask0x3c9fc80   98 NNVTMCPPMTsMask0x3c9fa80 
    nrpo(  70962     2     0     2 )            NNVTMCPPMT_PMT_20inch_log_phys0x3c9fe80                 NNVTMCPPMT_PMT_20inch_log0x3caec40  102 NNVTMCPPMT_PMT_20inch_pmt_solid0x3ca9320 
    nrpo(  70963     2     0     3 )           NNVTMCPPMT_PMT_20inch_body_phys0x3caefa0            NNVTMCPPMT_PMT_20inch_body_log0x3caeb60  101 NNVTMCPPMT_PMT_20inch_body_solid0x3cad240 
    nrpo(  70964     2     0     4 )         NNVTMCPPMT_PMT_20inch_inner1_phys0x3caf030          NNVTMCPPMT_PMT_20inch_inner1_log0x3caed60   99 NNVTMCPPMT_PMT_20inch_inner1_solid_1_Ellipsoid0x3503950 
    nrpo(  70965     2     0     5 )         NNVTMCPPMT_PMT_20inch_inner2_phys0x3caf0f0          NNVTMCPPMT_PMT_20inch_inner2_log0x3caee80  100 NNVTMCPPMT_PMT_20inch_inner2_solid0x3cae8f0 
    nrpo(  70966     3     0     0 )                    pLPMT_Hamamatsu_R128600x3cbbae0               HamamatsuR12860lMaskVirtual0x3c9a5c0  109 HamamatsuR12860sMask_virtual0x3c99fb0 
    nrpo(  70967     3     0     1 )                      HamamatsuR12860pMask0x3c9b320                      HamamatsuR12860lMask0x3c9b1a0  104 HamamatsuR12860sMask0x3c9afa0 
    nrpo(  70968     3     0     2 )       HamamatsuR12860_PMT_20inch_log_phys0x3c9b3b0            HamamatsuR12860_PMT_20inch_log0x3c93920  108 HamamatsuR12860_PMT_20inch_pmt_solid_1_90x3cb68e0 
    nrpo(  70969     3     0     3 )      HamamatsuR12860_PMT_20inch_body_phys0x345b3c0       HamamatsuR12860_PMT_20inch_body_log0x3c93830  107 HamamatsuR12860_PMT_20inch_body_solid_1_90x3ca7680 
    nrpo(  70970     3     0     4 )    HamamatsuR12860_PMT_20inch_inner1_phys0x3c94040     HamamatsuR12860_PMT_20inch_inner1_log0x345b160  105 HamamatsuR12860_PMT_20inch_inner1_solid_I0x3c96fa0 
    nrpo(  70971     3     0     5 )    HamamatsuR12860_PMT_20inch_inner2_phys0x3c94100     HamamatsuR12860_PMT_20inch_inner2_log0x345b290  106 HamamatsuR12860_PMT_20inch_inner2_solid_1_90x3c93610 
    nrpo( 304636     4     0     0 )     mask_PMT_20inch_vetolMaskVirtual_phys0x4433460          mask_PMT_20inch_vetolMaskVirtual0x3ca10e0  126 mask_PMT_20inch_vetosMask_virtual0x3ca0a80 
    nrpo( 304637     4     0     1 )                 mask_PMT_20inch_vetopMask0x3ca1e40                 mask_PMT_20inch_vetolMask0x3ca1cb0  121 mask_PMT_20inch_vetosMask0x3ca1aa0 
    nrpo( 304638     4     0     2 )                  PMT_20inch_veto_log_phys0x3ca5fa0                       PMT_20inch_veto_log0x3ca5470  125 PMT_20inch_veto_pmt_solid_1_20x3ca38b0 
    nrpo( 304639     4     0     3 )                 PMT_20inch_veto_body_phys0x3ca57a0                  PMT_20inch_veto_body_log0x3ca5360  124 PMT_20inch_veto_body_solid_1_20x3ca4230 
    nrpo( 304640     4     0     4 )               PMT_20inch_veto_inner1_phys0x3ca5820                PMT_20inch_veto_inner1_log0x3ca5580  122 PMT_20inch_veto_inner1_solid0x3ca4f10 
    nrpo( 304641     4     0     5 )               PMT_20inch_veto_inner2_phys0x3ca58d0                PMT_20inch_veto_inner2_log0x3ca5690  123 PMT_20inch_veto_inner2_solid0x3ca5130 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 


Collect virtual mask names::

   103 NNVTMCPPMTsMask_virtual0x3cb3b40
   109 HamamatsuR12860sMask_virtual0x3c99fb0  
   126 mask_PMT_20inch_vetosMask_virtual0x3ca0a80 

Check with GMeshLibTest::

    GMeshLibTest 103 109 126 

    epsilon:issues blyth$ GMeshLibTest 103 109 126 
    NNVTMCPPMTsMask_virtual0x3cb3b40
    HamamatsuR12860sMask_virtual0x3c99fb0
    mask_PMT_20inch_vetosMask_virtual0x3ca0a80
    epsilon:issues blyth$ 

    epsilon:issues blyth$ GMeshLibTest NNVTMCPPMTsMask_virtual0x HamamatsuR12860sMask_virtual0x mask_PMT_20inch_vetosMask_virtual0x
    103
    109
    126


So the option to use is::

    --skipsolidname NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x
   
Check that with::

    OpticksDbg=INFO GGeoTest --skipsolidname NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x

    epsilon:ggeo blyth$ OpticksDbg=INFO GGeoTest --skipsolidname NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x
    PLOG::EnvLevel adjusting loglevel by envvar   key OpticksDbg level INFO fallback DEBUG
    2021-04-23 14:42:36.618 INFO  [29545435] [OpticksDbg::postconfigure@229]  spec ~0 SBit::PosString(bitfield,',',true) ~0p,
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@179] [
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@197]  midx  103 sn [NNVTMCPPMTsMask_virtual0x]
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@197]  midx  109 sn [HamamatsuR12860sMask_virtual0x]
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@197]  midx  126 sn [mask_PMT_20inch_vetosMask_virtual0x]
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@208]  --skipsolidname NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x solidname.size 3 soidx.size 3
    2021-04-23 14:42:38.005 INFO  [29545435] [OpticksDbg::postgeometry@215] ]
    2021-04-23 14:42:38.309 INFO  [29545435] [*GParts::Create@244]  SKIPPING SOLID FROM ANALYTIC GEOMETRY VIA COMMANDLINE OPTION  i 0 num_pt 6 lvIdx 103 deferredcsgskiplv 0 skipsolidname 1
    2021-04-23 14:42:38.310 INFO  [29545435] [*GParts::Create@244]  SKIPPING SOLID FROM ANALYTIC GEOMETRY VIA COMMANDLINE OPTION  i 0 num_pt 6 lvIdx 109 deferredcsgskiplv 0 skipsolidname 1
    2021-04-23 14:42:38.311 INFO  [29545435] [*GParts::Create@244]  SKIPPING SOLID FROM ANALYTIC GEOMETRY VIA COMMANDLINE OPTION  i 0 num_pt 6 lvIdx 126 deferredcsgskiplv 0 skipsolidname 1
    2021-04-23 14:42:38.317 INFO  [29545435] [test_getNumMeshes@326]  num_meshes_ggeo 137 num_meshes_sgeo 137
    epsilon:ggeo blyth$ 







