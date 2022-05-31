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

* TODO : virtual skipping with more robust interface than ELV integers

* problem with ELV integers is that they silently change their meanings as the geometry changes, 
  so need to instead accept names which at least can fail loudly when the names can 
  no longer be converted into the integers


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




