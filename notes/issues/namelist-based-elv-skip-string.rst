namelist-based-elv-skip-string
===================================

* from :doc:`cxsim-shakedown`


* the other aspect is that such water/water virtuals should be skipped anyhow
  from Opticks geometry as they are just there for Geant4 performance reasons 

* the virtuals inevitably slow down Opticks and clog up step recording 
  with uninteresting steps

  * HMM what about step comparison with Geant4 ? 
  * ELV can dynamically skip solids (loading from same full geometry CSGFoundry)
    so can easily reinstate the virtuals without needing a different geocache  
    if want to do step to step comparison

* DONE : virtual skipping with more robust interface than ELV integers

* problem with ELV integers is that they silently change their meanings as the geometry changes, 
  so need to instead accept names which at least can fail loudly when the names can 
  no longer be converted into the integers


ELV not such a good way for  Water///Water skipping
------------------------------------------------------

Am forming the opinion that ELV is not the appropriate way to do more long lived skips
like virtual Water///Water skipping. Better to use cxskiplv and do the skips during initial CSGFoundry 
creation. 



cxskiplv
----------

::

    226 unsigned CSG_GGeo_Convert::CountSolidPrim( const GParts* comp )
    227 {
    228     unsigned solidPrim = 0 ;
    229     unsigned numPrim = comp->getNumPrim();
    230     for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++)
    231     {
    232         unsigned meshIdx   = comp->getMeshIndex(primIdx);   // from idxBuffer aka lvIdx 
    233         bool cxskip = SGeoConfig::IsCXSkipLV(meshIdx);
    234         if(cxskip)
    235         {
    236             LOG(LEVEL) << " cxskip meshIdx " << meshIdx  ;
    237         }
    238         else
    239         {
    240             solidPrim += 1 ;
    241         }
    242     }
    243     LOG(LEVEL) << " numPrim " << numPrim  << " solidPrim " << solidPrim ;
    244     return solidPrim ;
    245 }




Where meshnames come from and how are skips handled ?
------------------------------------------------------------


:doc:`primIdx-and-skips`




Seemed to apply with the cut hatbox removed in the simtrace plot of intersects but the virtual name is still in simtrace legend ?
------------------------------------------------------------------------------------------------------------------------------------

* probably this is mis-naming because the geocache meshname.txt is not changed by the dynamic selection, 
  but that is what the python name dict is using ?

  * hmm: this favors non-dynamic selection  

* TODO: check how CSGSelect impacts meshnames and take look at new meshname.txt 

From cxsim.sh before the ELVSelection the virtuals are there::

    In [29]: list(map(lambda _:pd[_], prim_(r)[0] ))                                                                                                                                                                                          
    Out[29]: 
    ['sWorld0x577dcb0',
     'sTarget0x5829390',
     'sTarget0x5829390',
     'sAcrylic0x5828d70',
     'NNVTMCPPMTsMask_virtual0x5f5f0e0',
     'NNVTMCPPMTsMask0x5f60190',
     'NNVTMCPPMTsMask0x5f60190',
     'NNVTMCPPMTTail0x5f614d0',
     'NNVTMCPPMTTail0x5f614d0',
     'sWorld0x577dcb0']



::

    2022-05-31 23:12:39.747 INFO  [233322] [SGeoConfig::GeometrySpecificSetup@115]  JUNO_detect 1
    2022-05-31 23:12:39.747 INFO  [233322] [CSGFoundry::CopySelect@2188]    -t117,110,134 141 : 111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110111111011111111111111110111111
    CSGFoundry::descELV elv.num_bits 141 include 138 exclude 3
    INCLUDE:138

      0:  0:sTopRock_domeAir0x578c250
      1:  1:sTopRock_dome0x578bd00
      2:  2:sDomeRockBox0x578b9f0
      3:  3:PoolCoversub0x578d190
      4:  4:Upper_LS_tube0x71a2df0
      5:  5:Upper_Steel_tube0x71a2ee0
      6:  6:Upper_Tyvek_tube0x71a2ff0
     ...
      7:  7:Upper_Chimney0x71a2d00
      8:  8:sBar0x71a8870
      9:  9:sBar0x71a8700
     10: 10:sPanelTape0x71a8590
     11: 11:sPanel0x71a8290
     12: 12:sPlane0x71a80b0
     13: 13:sWall0x71a8030
     14: 14:sAirTT0x71a6ba0
     15: 15:sExpHall0x578ccd0
     16: 16:sExpRockBox0x578c5e0
     17: 17:sTopRock0x578b880
     18: 18:GLw1.up10_up11_FlangeI_Web_FlangeII0x57929b0
     19: 19:GLw1.up09_up10_FlangeI_Web_FlangeII0x5793d30
     ...
     88: 88:ZC2.A03_A03_FlangeI_Web_FlangeII0x57c0f70
     89: 89:ZC2.A05_A05_FlangeI_Web_FlangeII0x57c3550
     90: 90:solidSJCLSanchor0x5961ce0
     91: 91:solidSJFixture0x5966120
     92: 92:solidSJReceiver0x5964730
     93: -1:solidSJReceiverFastern0x5968d50
     94: 94:sTarget0x5829390
     95: 95:sAcrylic0x5828d70
     96: 96:sStrut0x582be60
     97: 97:sStrut0x587f930
     98: 98:sStrutBallhead0x5852e20
     99: -1:uni10x58327d0
    100:100:base_steel0x58d2a50
    101:101:uni_acrylic10x597b870
    102:102:solidXJanchor0x59328e0
    103:103:solidXJfixture0x595eb40
    104:104:HamamatsuR12860sMask0x5f51a40
    105:105:HamamatsuR12860Tail0x5f52eb0
    106:106:HamamatsuR12860_PMT_20inch_inner1_solid_I0x5f39240
    107:107:HamamatsuR12860_PMT_20inch_inner2_solid_1_40x5f4b5c0
    108:108:HamamatsuR12860_PMT_20inch_body_solid_1_40x5f45b70
    109:109:HamamatsuR12860_PMT_20inch_pmt_solid_1_40x57f1df0
    111:111:NNVTMCPPMTsMask0x5f60190
    112:112:NNVTMCPPMTTail0x5f614d0
    113:113:NNVTMCPPMT_PMT_20inch_inner1_solid_head0x5f56e60
    114:114:NNVTMCPPMT_PMT_20inch_inner2_solid_head0x5f5c800
    115:115:NNVTMCPPMT_PMT_20inch_body_solid_head0x5f5a9d0
    116:116:NNVTMCPPMT_PMT_20inch_pmt_solid_head0x5f58840
    118:118:PMT_3inch_inner1_solid_ell_helper0x66e54d0
    119:119:PMT_3inch_inner2_solid_ell_helper0x66e5570
    120:120:PMT_3inch_body_solid_ell_ell_helper0x66e5430
    121:121:PMT_3inch_cntr_solid0x66e5640
    122:122:PMT_3inch_pmt_solid0x66e51d0
    123:123:sChimneyAcrylic0x71a5510
    124:124:sChimneyLS0x71a56f0
    125:125:sChimneySteel0x71a58d0
    126:126:sWaterTube0x71a5330
    127:127:sInnerWater0x5828750
    128:128:sReflectorInCD0x5828190
    129:129:mask_PMT_20inch_vetosMask0x5f63630
    130:130:PMT_20inch_veto_inner1_solid0x5f66ad0
    131:131:PMT_20inch_veto_inner2_solid0x5f66cc0
    132:132:PMT_20inch_veto_body_solid_1_20x5f65e80
    133:133:PMT_20inch_veto_pmt_solid_1_20x5f65570
    135:135:sOuterWaterPool0x57923b0
    136:136:sPoolLining0x5791ca0
    137:137:sBottomRock0x578d8c0
    138:138:sWorld0x577dcb0
    139: -1:solidSJReceiverFastern0x5968d50
    140: -1:uni10x58327d0
    EXCLUDE:3

    110:110:HamamatsuR12860sMask_virtual0x5f50520
    117:117:NNVTMCPPMTsMask_virtual0x5f5f0e0
    134:134:mask_PMT_20inch_vetosMask_virtual0x5f62620





WIP : implemented mostly in SName
-----------------------------------------

* HMM: this is very detector specific, so how/where to config/invoke it 
* HMM: do i want to auto skip solids with _virtual0x in their names ?

  * ie make "_virtual0x" the input skip string for JUNO ?
  * certainly convenient : perhaos a little too automated 

  * ALSO will likely be more solids to skip, so probably best to list the full name starts
    in the skipstring that is the input to forming the ELV

  * this forces awareness of name changes as the ELV forming will assert with stale names 

  * could config via SGeoConfig statics so the opticks setup JUNO 
    code can call the method to setting the skip string 


::

    epsilon:sysrap blyth$ QTYPE=C SNameTest _virtual0x
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                                       _virtual0x count   0 idx  -1
     findIndices                                                                       _virtual0x idxs.size   3 SName::QTypeLabel CONTAIN
    descIndices
     110 : NNVTMCPPMTsMask_virtual0x5f5f900
     117 : HamamatsuR12860sMask_virtual0x5f50d40
     134 : mask_PMT_20inch_vetosMask_virtual0x5f62e40

    SName::ELVString:[t110,117,134]
    test_get_ELV_skipString contain [_virtual0x] elv [t110,117,134]
    epsilon:sysrap blyth$ 


ELV review
-----------------

  
::

    epsilon:opticks blyth$ opticks-f ELV
    ./CSGOptiX/cxr_scan.sh:        ELV=$e ./$script.sh $*
    ./CSGOptiX/cxr_overview.sh:export ELV=${ELV:-$elv}
    ./CSGOptiX/cxr_overview.sh:export NAMEPREFIX=cxr_overview_emm_${EMM}_elv_${ELV}_moi_      # MOI gets appended by the executable
    ./CSG/CSGFoundry.h:    const std::string descELV(const SBitSet* elv) const ; 
    ./CSG/CSGPrimTest.sh:    ELV=103 ./CSGPrimTest.sh 
    ./CSG/CSGPrimTest.sh:         ELV SBitSet prim selection based on meshIdx with CSGCopy::Select  
    ./CSG/tests/CSGCopyTest.cc:    const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", "t" ); 
    ./CSG/tests/CSGCopyTest.cc:    LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 
    ./CSG/CSGFoundry.cc:const std::string CSGFoundry::descELV(const SBitSet* elv) const 
    ./CSG/CSGFoundry.cc:    ss << "CSGFoundry::descELV" 
    ./CSG/CSGFoundry.cc:    const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", nullptr ); 
    ./CSG/CSGFoundry.cc:    LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 
    ./sysrap/SBitSet.cc:void SBitSet::set_label(const char* label_) // eg ELV or EMM 


::

    2073 /**
    2074 CSGFoundry::Load
    2075 -------------------
    2076 
    2077 This argumentless Load method is special, unlike other methods 
    2078 it provides dynamic prim selection based in the ELV envvar which uses
    2079 CSGFoundry::CopySelect to dynamically create a CSGFoundry based
    2080 on the elv SBitSet
    2081 
    2082 **/
    2083 CSGFoundry* CSGFoundry::Load() // static
    2084 {   
    2085     CSGFoundry* src = CSGFoundry::Load_() ;
    2086     if(src == nullptr) return nullptr ; 
    2087     const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", nullptr );
    2088     CSGFoundry* dst = elv ? CSGFoundry::CopySelect(src, elv) : src  ;
    2089     return dst ;
    2090 }



::

    epsilon:CSG blyth$ ELV=103 ./CSGPrimTest.sh
    2022-05-31 10:20:11.213 INFO  [11435221] [*CSGFoundry::CopySelect@2090]  ELV       103 141 : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000
    CSGFoundry::descELV elv.num_bits 141 include 1 exclude 140
    INCLUDE:1

    103:solidXJfixture0x595eb40
    EXCLUDE:140

      0:sTopRock_domeAir0x578c250
      1:sTopRock_dome0x578bd00
      2:sDomeRockBox0x578b9f0
      3:PoolCoversub0x578d190
      4:Upper_LS_tube0x71a2df0
      5:Upper_Steel_tube0x71a2ee0
      6:Upper_Tyvek_tube0x71a2ff0
      7:Upper_Chimney0x71a2d00
      8:sBar0x71a8870
      9:sBar0x71a8700
     10:sPanelTape0x71a8590
     11:sPanel0x71a8290
     12:sPlane0x71a80b0
     13:sWall0x71a8030
     14:sAirTT0x71a6ba0
     15:sExpHall0x578ccd0
     16:sExpRockBox0x578c5e0
     17:sTopRock0x578b880
     18:GLw1.up10_up11_FlangeI_Web_FlangeII0x57929b0
     19:GLw1.up09_up10_FlangeI_Web_FlangeII0x5793d30
     20:GLw1.up08_up09_FlangeI_Web_FlangeII0x57966a0




