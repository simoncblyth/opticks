stree_factorize_steering_to_place_selected_lv_into_triangulated_CSGSolid
==========================================================================

Requirement
------------

Need user control indicating that certain named solids must be triangulated.  
HMM for solids that are not repeated many times could have a single 
triangulated global CSGSolid to gather all the triangulated solids. 
OR could have separate CSGSolid for each that needs to be triangulated.  

HMM: in current JUNO geom I expect all that need to be triangulated 
will be in the non-instanced remainder geometry

There are other lvid name selections, can follow those for selection mechanics. 



HMM : lvid solid name selections following SGeoConfig::IsCXSkipLV(int lv) is too late to steer stree factorization 
---------------------------------------------------------------------------------------------------------------------

SO : NEED TO SO SOMETHING SIMILAR AT U4Tree/stree LEVEL 


HMM: need the SName to convert solid names into lvid indices.
that is maybe not early enough to steer factorization::

    void SGeoConfig::GeometrySpecificSetup(const SName* id) 

::

    2915 CSGFoundry* CSGFoundry::Load() // static
    2916 {
    2917     SProf::Add("CSGFoundry__Load_HEAD");
    2918 
    2919     LOG(LEVEL) << "[ argumentless " ;
    2920     CSGFoundry* src = CSGFoundry::Load_() ;
    2921     if(src == nullptr) return nullptr ;
    2922 
    2923     SGeoConfig::GeometrySpecificSetup(src->id);
    2924 
    2925     const SBitSet* elv = ELV(src->id);
    2926     CSGFoundry* dst = elv ? CSGFoundry::CopySelect(src, elv) : src  ;
    2927 
    2928     if( elv != nullptr && Load_saveAlt)
    2929     {
    2930         LOG(error) << " non-standard dynamic selection CSGFoundry_Load_saveAlt " ;
    2931         dst->saveAlt() ;
    2932     }
    2933 
    2934     AfterLoadOrCreate();
    2935 
    2936     LOG(LEVEL) << "] argumentless " ;
    2937     SProf::Add("CSGFoundry__Load_TAIL");
    2938     return dst ;
    2939 }




cxr_min.sh TRIMESH
---------------------

This is selection of solids to be triangulated after the factorization is 
done. It does not influence the factorization.  

::

    147 if [ -n "$TRIMESH" ]; then
    148 
    149    #trimesh=2923:sWorld
    150    #trimesh=5:PMT_3inch_pmt_solid
    151    #trimesh=9:NNVTMCPPMTsMask_virtual
    152    #trimesh=12:HamamatsuR12860sMask_virtual
    153    #trimesh=6:mask_PMT_20inch_vetosMask_virtual
    154    #trimesh=1:sStrutBallhead
    155    #trimesh=1:base_steel
    156    #trimesh=1:uni_acrylic1
    157    #trimesh=130:sPanel
    158 
    159    #trimesh=4:VACUUM_solid
    160 
    161    #trimesh=3:Rock_solid 
    162 
    163    trimesh=28:World_solid
    164 
    165    #trimesh=2:VACUUM_solid 
    166 
    167    export OPTICKS_SOLID_TRIMESH=$trimesh
    168 fi


This is selection of CSGSolid by mmlabel after the factorization is done. 


::

    [blyth@localhost sysrap]$ opticks-f SolidTrimesh 
    ./CSG/CSGFoundry.cc:CSGFoundry::isSolidTrimesh
    ./CSG/CSGFoundry.cc:bool CSGFoundry::isSolidTrimesh(int gas_idx) const 
    ./CSG/CSGFoundry.cc:    const char* ls = SGeoConfig::SolidTrimesh() ; 
    ./CSG/CSGFoundry.h:    bool isSolidTrimesh(int gas_idx) const ;  // see SGeoConfig::SolidTrimesh 
    ./CSGOptiX/SBT.cc:    bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:    bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./CSGOptiX/SBT.cc:        bool trimesh = foundry->isSolidTrimesh(gas_idx); 
    ./sysrap/SGeoConfig.cc:const char* SGeoConfig::_SolidTrimesh   = ssys::getenvvar(kSolidTrimesh, nullptr ); 
    ./sysrap/SGeoConfig.cc:void SGeoConfig::SetSolidTrimesh(  const char* st){  _SolidTrimesh   = st ? strdup(st) : nullptr ; }
    ./sysrap/SGeoConfig.cc:const char* SGeoConfig::SolidTrimesh(){   return _SolidTrimesh ; }
    ./sysrap/SGeoConfig.cc:    ss << std::setw(25) << kSolidTrimesh     << " : " << ( _SolidTrimesh   ? _SolidTrimesh   : "-" ) << std::endl ;    
    ./sysrap/SGeoConfig.hh:    static constexpr const char* kSolidTrimesh   = "OPTICKS_SOLID_TRIMESH" ; 
    ./sysrap/SGeoConfig.hh:    static constexpr const char* kSolidTrimesh_desc = "CSGFoundry comma delimited list of CSGSolid for Trimesh geometry" ; 
    ./sysrap/SGeoConfig.hh:    static const char* _SolidTrimesh ;   
    ./sysrap/SGeoConfig.hh:    static const char* SolidTrimesh(); 
    ./sysrap/SGeoConfig.hh:    static void SetSolidTrimesh(   const char* ss ); 
    [blyth@localhost opticks]$ 

::

     244 /**
     245 CSGFoundry::isSolidTrimesh
     246 ----------------------------
     247 
     248 Normally returns false indicating to use analytic solid setup, 
     249 can arrange to return true for some CSGSolid using envvar 
     250 with comma delimited mmlabel indicating to use approximate
     251 triangulated geometry for those solids::
     252 
     253    export OPTICKS_SOLID_TRIMESH=1:sStrutBallhead,1:base_steel
     254 
     255 **/
     256 bool CSGFoundry::isSolidTrimesh(int gas_idx) const
     257 {
     258     const char* ls = SGeoConfig::SolidTrimesh() ;
     259     if(ls == nullptr) return false ;
     260     return SLabel::IsIdxLabelListed( mmlabel, gas_idx, ls, ',' );
     261 }




SSim/stree/soname_names.txt
----------------------------

Lists solid names





