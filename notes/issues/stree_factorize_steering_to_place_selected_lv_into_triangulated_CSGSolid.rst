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

* actually cannot reuse those directly, as need to operate earlier within stree.h  



CSGFoundry::id (SName) where does it come from ?
--------------------------------------------------

::

    SName*    id ;   // holds the meshname vector of G4VSolid names


::

     095 CSGFoundry::CSGFoundry()
      96     :
      97     d_prim(nullptr),
      98     d_node(nullptr),
      99     d_plan(nullptr),
     100     d_itra(nullptr),
     101     sim(SSim::Get()),
     102     import(new CSGImport(this)),
     103     id(new SName(meshname)),   // SName takes a reference of the meshname vector of strings 
     104     target(new CSGTarget(this)),
     105     maker(new CSGMaker(this)),


::

     62 void CSGImport::importNames()
     63 {
     64     st->get_mmlabel( fd->mmlabel);
     65     st->get_meshname(fd->meshname);
     66 }


CSGFoundry::meshname comes from stree::soname with stree::Name applied 
which strips the 0x tail if present::

    1701 inline void stree::get_meshname( std::vector<std::string>& names) const
    1702 {
    1703     assert( names.size() == 0 );
    1704     for(unsigned i=0 ; i < soname.size() ; i++) names.push_back( Name(soname[i],true) );
    1705 }

    1691 inline std::string stree::Name( const std::string& name, bool strip ) // static
    1692 {
    1693     return strip ? sstr::StripTail(name, "0x") : name ;
    1694 }


With original geometry no stripping is needed::

    [blyth@localhost stree]$ cat /home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/stree/soname_names.txt
    sTopRock_domeAir
    sTopRock_dome
    sDomeRockBox
    PoolCoversub
    Upper_LS_tube
    Upper_Steel_tube
    Upper_Tyvek_tube
    Upper_Chimney



DONE : added sstr::HasTail that works on vectors of names, so can form SName at stree level when not operating from GDML so no stripping needed
--------------------------------------------------------------------------------------------------------------------------------------------------

At what juncture does stree have all the solid names ?



"stree__force_triangulate_solid" ? How to use a list of solid names to be force triangulated
-----------------------------------------------------------------------------------------------

::

    2992 inline void stree::collectRemainderNodes()
    2993 {
    2994     assert( rem.size() == 0u );
    2995     for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    2996     {
    2997         const snode& nd = nds[nidx] ;
    2998         assert( nd.index == nidx );
    2999         if( nd.repeat_index == 0 ) rem.push_back(nd) ;
    3000     }
    3001     if(level>0) std::cout << "stree::collectRemainderNodes rem.size " << rem.size() << std::endl ;
    3002 }


* nd.lvid 
 

DONE : stree::is_force_triangulate_solid(int lvid)
----------------------------------------------------

1. convert envvar list of solid names into list if lvid using the stree list of uniqued solid names


DONE : splitting remainder into analytic *rem* and triangulated *tri* vectors 
-------------------------------------------------------------------------------------------------------------------------

TODO: add persisting of the *tri* snode
-------------------------------------------

TODO : add creation of CSGSolid from the *tri* snode
---------------------------------------------------------


stree::is_force_triangulate_solid ? Where to use a list of solid names to be force triangulated
-------------------------------------------------------------------------------------------------

* Within stree::factorize/stree::collectRemainderNodes is natural place as the list of solid names is available from the U4Tree ctor 
* can initially require(or assert) forced triangle solids to be in the remainder : that will be the normal case  


::

     202 inline U4Tree* U4Tree::Create(
     203     stree* st,
     204     const G4VPhysicalVolume* const top,
     205     U4SensorIdentifier* sid
     206     )
     207 {
     208     if(st->level > 0) std::cout << "[ U4Tree::Create " << std::endl ;
     209 
     210     U4Tree* tree = new U4Tree(st, top, sid ) ;
     211 
     212     st->factorize();
     213 
     214     tree->identifySensitive();
     215 
     216     st->add_inst();
     217 
     218     if(st->level > 0) std::cout << "] U4Tree::Create " << std::endl ;
     219 
     220     st->postcreate() ;
     221 
     222     return tree ;
     223 }


::

     250 inline void U4Tree::init()
     251 {
     252     if(top == nullptr) return ;
     253 
     254     initRayleigh();
     255     initMaterials();
     256     initMaterials_NoRINDEX();
     257 
     258     initScint();
     259 
     260     initSurfaces();
     261     initSolids();
     262     initNodes();
     263     initSurfaces_Serialize();
     264 
     265     initStandard();
     266 
     267     std::cout << "U4Tree::init " <<  desc() << std::endl;
     268 
     269 }



U4Tree::initSolids_Keys already handles solid name stripping and uniqing 
-----------------------------------------------------------------------------

::

    523 inline void U4Tree::initSolids()
    524 {
    525     initSolids_r(top);
    526     initSolids_Keys();
    527     initSolids_Mesh();
    528 }
    529 
    530 /**
    531 U4Tree::initSolids_Keys
    532 ------------------------
    533 
    534 The st->soname_raw which may have 0x suffixes are
    535 tail stripped and if needed uniqued wiyj _0 _1 suffix
    536 to form st->soname
    537 
    538 **/
    539 
    540 inline void U4Tree::initSolids_Keys()
    541 {
    542     sstr::StripTail_Unique( st->soname, st->soname_raw, "0x" );
    543     assert( st->soname.size() == st->soname_raw.size() );
    544 }







OPTICKS_SOLID_TRIMESH SGeoConfig::Trimesh CSGFoundry::isSolidTrimesh
-------------------------------------------------------------------------

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
     262 







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





