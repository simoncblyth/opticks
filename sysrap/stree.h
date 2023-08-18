#pragma once
/**
stree.h : minimal representation of the structural geometry tree
=====================================================================

This is exploring a minimal approach to geometry translation

* see also u4/U4Tree.h that populates stree.h from traversals of Geant4 volumes. 
* stree.h is part of the attempt to replace lots of GGeo code, notably: GInstancer.cc GNode.cc 

WIP : stree fold getting messy
--------------------------------

* DONE : split off mat/sur/bnd into sstandard 
* TODO : split off structural transforms, nodes etc.. into subfold 


Lifecycle
------------

* Canonical stree instance is SSim member instanciated by SSim::SSim 
* stree is populated by U4Tree::Create


SSim+stree vs CSGFoundry 
--------------------------

Some duplication between these is inevitable, however they have 
different objectives:

* *SSim+stree* aims to collect and persist all needed info from Geant4 
* *CSGFoundry* aims to prepare the subset that needs to be uploaded to GPU

  * narrowing to float is something that could be done when going from stree->CSGFoundry 


Users of stree.h
-------------------

u4/U4Tree.h
    heavy lifting of populating stree.h 

CSG_GGeo/CSG_GGeo_Convert.cc
    stree.h/tree member obtained from "SSim::Get()->get_tree()"
    tree used from CSG_GGeo_Convert::addInstances to stree::lookup_sensor_identifier
    the sensor_id and sensor_index are incorporated into the CSGFoundry instances 
    (so this usage is "precache")

    NB : THIS IS UNHOLY MIX OF OLD AND NEW : TO BE REPLACED

ggeo/tests/GGeoLoadFromDirTest.cc
    dev of the interim stree GGeo integration for sensor info

ggeo/GGeo.cc
    GGeo:m_tree with setTree/getTree : but treated as foreign member, only GGeo::save saves it 
    this m_tree was used for debugging a now resolved discrepancy between X4+GGeo and U4Test 
    transforms : that is suspected but not confirmed to have been caused by a 
    stree parent pointer bug 

extg4/X4PhysicalVolume.cc
    X4PhysicalVolume::convertStructure creates stree.h and setTree into GGeo 
    X4PhysicalVolume::convertStructure_r collects snode.h and transforms into the GGeo/stree 

sysrap/SBnd.h
    SBnd::FillMaterialLine uses the boundary specs to convert stree.h mtname into mtline 
    for texture lookups and material index to line mappings needed at genstep collection



Find users of stree.h
------------------------

::

    epsilon:opticks blyth$ opticks-fl stree.h | grep -v stree
    ./CSG/CSGTarget.cc
    ./CSG/tests/CSGFoundryLoadTest.cc
    ./extg4/X4PhysicalVolume.cc
    ./sysrap/CMakeLists.txt
    ./sysrap/SBnd.h
    ./sysrap/sphit.h
    ./sysrap/sframe.h
    ./sysrap/SSim.cc
    ./ggeo/GGeo.cc
    ./ggeo/tests/GGeoLoadFromDirTest.cc
    ./u4/U4Tree.h
    ./CSG_GGeo/CSG_GGeo_Convert.cc



TODO Overview
-------------------

* maintain correspondence between source nodes and destination nodes thru the factorization
* triplet_identity ?
* transform rebase
* solids nsphere ncone etc... into a more uniform direct to CSGPrim approach 



WIP: standardize to using NPFold.h serialize/import pattern 


Optimization ideas
--------------------

Work out how the time is split and consider pruning, 
not everything needs to be persisted eg 

gtd
inst_f4
iinst_f4 

POSSIBLY : make saving full nds nodes optional as the 
factorization might be made to replace the full info ? 




TODO : mapping from "factorized" instances back to origin PV and vice-versa 
-----------------------------------------------------------------------------

Excluding the remainder, the instances each correspond to contiguous ranges of nidx.
So can return back to the origin pv volumes using "U4Tree::get_pv(int nidx)" so long  
as store the outer nidx and number of nodes within the instances. 

Actually as all factorized instances of each type will have the same number of 
subtree nodes it makes no sense to duplicate that info : better to store the node counts 
in an "srepeat" instance held within stree.h and just store the first nidx within the instances. 
The natural place for this is within the spare 4th column of the instance transforms.    

TODO: modify sqat4.h::setIdentity to store this nidx and populate it 
HMM : actually better to defer until do stree->CSGFoundry direct translation  

For going in the other direction "int U4Tree::get_nidx(const G4VPhysicalVolume* pv) const" 
provides the nidx of a pv. Then can search the factorized instances for that nidx 
using the start nidx of each instance and number of nidx from the "srepeat".
When the query nidx is not found within the instances it must be from the remainder. 

HMM: it would be faster to include the triplet_identity within the snode then 
can avoid the searching

* need API to jump to the object within the CF model given the triplet_identity   

The natural place to keep map back info is within the instance transforms. 


mapping for the remainder non-instanced volumes
-------------------------------------------------

For the global remainder "instance" things are not so straightforward as there is only 
one of them with an identity transform and the nodes within it are not contiguous, they 
are what is left when all the repeated subtrees have been removed : so it will 
start from root nidx:0 and will have lots of gaps. 

Actually the natural place to keep "map-back" info for the remainder 
is withing the CSGPrim.  Normally use of that is for identity purposes is restricted 
because tthe CSGPrim are references from all the instances but for the remainder
the CSGPrim are only referenced once (?). TODO: check this 

TODO: collect the nidx of the remainder into stree.h ?


static log level control
-------------------------

As stree (and also U4Tree) are header only they cannot easily 
have a static EnvLevel as initing a static in header only situation 
is complicated in C++11.  
With C++17 can supposedly do this easily with "inline static". See

https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library

Pragmatic workaround for runtime logging control is to 
rely on the "guardian" of stree, which is SSim, 
as that instanciates stree in can easily set a member. 

So the stree::set_level is invoked from SSim::init based on the envvar:: 

    export SSim__stree_level=0    # no logging   
    export SSim__stree_level=1    # minimal logging   
    export SSim__stree_level=2    # some logging   
    export SSim__stree_level=3    # verbose logging   

When SSim not in use can also use::

    export stree_level=1 


**/

#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "NP.hh"
#include "NPX.h"
#include "NPFold.h"

#include "ssys.h"
#include "sstr.h"
#include "scuda.h"
#include "snode.h"
#include "sdigest.h"
#include "sfreq.h"
#include "sstr.h"
#include "strid.h"
#include "sfactor.h"

#ifdef WITH_SND
#include "snd.hh"
#include "scsg.hh"
#else
#include "s_csg.h"
#include "sn.h"
#endif


#include "stra.h"
#include "sstandard.h"
#include "smatsur.h"
#include "snam.h"
#include "SBnd.h"


struct stree
{
    static constexpr const int MAXDEPTH = 15 ; // presentational only   
    static constexpr const int FREQ_CUT = 500 ;   // HMM GInstancer using 400   
    // subtree digests with less repeats than FREQ_CUT within the entire geometry 
    // are not regarded as repeats for instancing factorization purposes 

    static constexpr const char* BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ;  // default 
    static constexpr const char* RELDIR = "stree" ;

    static constexpr const char* NDS = "nds.npy" ;
    static constexpr const char* NDS_NOTE = "snode.h structural volume nodes" ;
    static constexpr const char* REM = "rem.npy" ;
    static constexpr const char* M2W = "m2w.npy" ;
    static constexpr const char* W2M = "w2m.npy" ;
    static constexpr const char* GTD = "gtd.npy" ;  // GGeo transform debug, populated in X4PhysicalVolume::convertStructure_r 
    static constexpr const char* TRS = "trs.npy" ;  // optional, when use save_trs
    static constexpr const char* MTNAME = "mtname.txt" ;
    static constexpr const char* MTNAME_NO_RINDEX = "mtname_no_rindex.txt" ;
    static constexpr const char* MTINDEX = "mtindex.npy" ;
    static constexpr const char* MTLINE = "mtline.npy" ;
    static constexpr const char* SUNAME = "suname.txt" ;
    static constexpr const char* IMPLICIT = "implicit.txt" ;

#ifdef DEBUG_IMPLICIT
    static constexpr const char* IMPLICIT_ISUR = "implicit_isur.npy" ;
    static constexpr const char* IMPLICIT_OSUR = "implicit_osur.npy" ;
#endif


    static constexpr const char* SUINDEX = "suindex.npy" ;

    static constexpr const char* SONAME = "soname.txt" ;
    static constexpr const char* CSG = "csg" ;
    static constexpr const char* _CSG = "_csg" ;
    static constexpr const char* SN = "sn" ;
    static constexpr const char* DIGS = "digs.txt" ;
    static constexpr const char* SUBS = "subs.txt" ;
    static constexpr const char* SUBS_FREQ = "subs_freq" ;
    static constexpr const char* FACTOR = "factor.npy" ;
    static constexpr const char* MATERIAL = "material" ;
    static constexpr const char* SURFACE = "surface" ;
    static constexpr const char* STANDARD = "standard" ;


    static constexpr const char* INST = "inst.npy" ; 
    static constexpr const char* IINST = "iinst.npy" ; 
    static constexpr const char* INST_F4 = "inst_f4.npy" ; 
    static constexpr const char* IINST_F4 = "iinst_f4.npy" ; 

    static constexpr const char* SENSOR_ID = "sensor_id.npy" ; 
    static constexpr const char* SENSOR_NAME = "sensor_name.npy" ; 
    static constexpr const char* INST_NIDX = "inst_nidx.npy" ; 

    int level ;                            // verbosity 

    std::vector<std::string> mtname ;       // unique material names
    std::vector<std::string> mtname_no_rindex ; 
    std::vector<int>         mtindex ;      // G4Material::GetIndex 0-based creation indices 
    std::vector<int>         mtline ;     
    std::map<int,int>        mtindex_to_mtline ;   // filled from mtindex and mtline with init_mtindex_to_mtline 
    // map not currently persisted (it could be using NPX.h)

    std::vector<std::string> suname ;       // surface names
    std::vector<int>         suindex ;      // HMM: is this needed, its just 0,1,2,...
    std::vector<int4>        vbd ; 
    std::vector<std::string> bdname ; 
    std::vector<std::string> implicit ;  // names of implicit surfaces

    std::vector<std::string> soname ;       // unique solid names

#ifdef WITH_SND
    std::vector<int>         solids ;       // snd idx 
#else
    std::vector<sn*>         solids ; 
#endif

    std::vector<glm::tmat4x4<double>> m2w ; // model2world transforms for all nodes
    std::vector<glm::tmat4x4<double>> w2m ; // world2model transforms for all nodes  
    std::vector<glm::tmat4x4<double>> gtd ; // GGeo Transform Debug, added from X4PhysicalVolume::convertStructure_r


    std::vector<snode> nds ;               // snode info for all structural nodes, the volumes
    std::vector<snode> rem ;               // selection of remainder nodes
    std::vector<std::string> digs ;        // per-node digest for all nodes  
    std::vector<std::string> subs ;        // subtree digest for all nodes
    std::vector<sfactor> factor ;          // small number of unique subtree factor, digest and freq  

    std::vector<int> sensor_id ;           // updated by reorderSensors
    unsigned sensor_count ; 
    std::vector<std::string> sensor_name ; 


    sfreq* subs_freq ;                     // occurence frequency of subtree digests in entire tree 
                                           // subs are collected in stree::classifySubtrees

#ifdef WITH_SND
    scsg*   csg ;                          // snd.hh based csg node trees of all solids from G4VSolid    
#else
    s_csg* _csg ;                          // sn.h based csg node trees
#endif

    sstandard* standard ;                  // mat/sur/bnd/bd/optical/wavelength/energy/rayleigh 

    NPFold* material ;   // material properties from G4 MPTs
    NPFold* surface ;    // surface properties from G4 MPTs, includes OpticalSurfaceName osn in metadata         


    std::vector<glm::tmat4x4<double>> inst ; 
    std::vector<glm::tmat4x4<float>>  inst_f4 ; 
    std::vector<glm::tmat4x4<double>> iinst ; 
    std::vector<glm::tmat4x4<float>>  iinst_f4 ; 

    std::vector<int>                  inst_nidx ; 

    stree();

    void init(); 
    void set_level(int level_); 

    std::string desc() const ;
    std::string desc_size(char div='\n') const ;
    std::string desc_vec() const ;
    std::string desc_sub(bool all=false) const ;
    std::string desc_sub(const char* sub) const ;


    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr );
    static std::string Digest(int lvid );

    template<typename T>
    static int GetPointerIndex( const std::vector<const T*>& vec, const T* obj) ; 
    template<typename T>
    static int GetValueIndex( const std::vector<T>& vec, const T& obj) ; 


    void get_children(std::vector<int>& children, int nidx) const ;   // immediate children
    void get_progeny( std::vector<int>& progeny, int nidx ) const ;   // recursively get children and all their children and so on... 
    std::string desc_progeny(int nidx) const ; 

    void traverse(int nidx=0) const ; 
    void traverse_r(int nidx, int depth, int sibdex) const ; 

    void reorderSensors(); 
    void reorderSensors_r(int nidx); 
    void get_sensor_id( std::vector<int>& arg_sensor_id ) const ; 

    void postcreate() const ; 

    std::string desc_sensor() const ; 
    int get_num_nd_sensor() const ; 
    void get_sensor_nidx( std::vector<int>& sensor_nidx ) const ; 

    std::string desc_sensor_nd(int edge) const ; 

    std::string desc_sensor_id(unsigned edge=10) const ; 
    static std::string DescSensor( const std::vector<int>& sensor_id, const std::vector<int>& sensor_idx, unsigned edge=10 ); 

    void lookup_sensor_identifier( 
         std::vector<int>& arg_sensor_identifier, 
         const std::vector<int>& arg_sensor_index, 
         bool one_based_index, bool verbose=false, unsigned edge=10 ) const ; 

    sfreq* make_progeny_freq(int nidx) const ; 
    sfreq* make_freq(const std::vector<int>& nodes ) const ; 

    int  find_lvid(const char* soname_, bool starting=true  ) const ; 
    void find_lvid_nodes_( std::vector<snode>& nodes, int lvid ) const ; 
    void find_lvid_nodes(  std::vector<int>& nodes, int lvid ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, const char* soname_, bool starting=true ) const ; 
    int  find_lvid_node( const char* q_soname, int ordinal ) const ; 
    int  find_lvid_node( const char* q_spec ) const ; // eg HamamatsuR12860sMask_virtual:0:1000

    void get_sub_sonames( std::vector<std::string>& sonames ) const ; 
    const char* get_sub_soname(const char* sub) const ; 

    static std::string Name( const std::string& name, bool strip ); 
    std::string get_lvid_soname(int lvid, bool strip ) const ; 
    void        get_meshname( std::vector<std::string>& names) const ;  // match CSGFoundry 
    void        get_mmlabel(  std::vector<std::string>& names) const ;  // match CSGFoundry 


    int         get_num_nodes() const ; 
    const char* get_soname(int nidx) const ; 
    const char* get_sub(   int nidx) const ; 
    int         get_depth( int nidx) const ; 
    int         get_parent(int nidx) const ; 
    int         get_lvid(  int nidx) const ; 
    int         get_copyno(int nidx) const ; 

    const snode* get_node(int nidx) const ; 
    const snode* get_parent_node(int nidx) const ; 
    bool         is_outer_node(int nidx) const ; 

    void         get_ancestors(std::vector<int>& ancestors, int nidx, bool local, std::ostream* out ) const ;
    std::string desc_ancestors(                             int nidx, bool local ) const ;


    void get_node_transform( glm::tmat4x4<double>& m2w_, glm::tmat4x4<double>& w2m_, int nidx ) const ; 
    void get_node_product(   
           glm::tmat4x4<double>& m2w_, 
           glm::tmat4x4<double>& w2m_, int nidx, bool local, bool reverse, std::ostream* out ) const ; 

    std::string desc_node_product(   glm::tmat4x4<double>& m2w_, glm::tmat4x4<double>& w2m_, int nidx, bool local, bool reverse ) const ; 


    template<typename N>
    void get_combined_transform(
             glm::tmat4x4<double>& t,
             glm::tmat4x4<double>& v,
             const snode& node,
             const N* nd,
             std::ostream* out
             ) const ;

    template<typename N>
    std::string desc_combined_transform(
             glm::tmat4x4<double>& t,
             glm::tmat4x4<double>& v,
             const snode& node,
             const N* nd
             ) const ;

    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ; 


    std::string subtree_digest( int nidx ) const ;
    static std::string depth_spacer(int depth); 

    std::string desc_node_(int nidx, const sfreq* sf ) const ;
    std::string desc_node(int nidx, bool show_sub=false) const ; 

    std::string desc_nodes( const std::vector<int>&   nn, int edgeitems=10) const ;
    std::string desc_nodes_(const std::vector<snode>& nn, int edgeitems=10) const ;
    std::string desc_solids() const ; 

    NP* make_trs() const ; 
    void save_trs(const char* fold) const ; 


    void save_( const char* fold ) const ;
    void save( const char* base, const char* reldir=RELDIR ) const ;
    NPFold* serialize() const ; 


    template<typename S, typename T>   // S:compound type T:atomic "transport" type
    static void ImportArray( std::vector<S>& vec, const NP* a ); 

    static void ImportNames( std::vector<std::string>& names, const NP* a ); 


    static stree* Load(const char* base=BASE , const char* reldir=RELDIR ); 

    int load( const char* base, const char* reldir=RELDIR );
    int load_( const char* fold );
    void import(const NPFold* fold); 


    static int Compare( const std::vector<int>& a, const std::vector<int>& b ) ; 
    static std::string Desc(const std::vector<int>& a, unsigned edgeitems=10 ) ; 


    void classifySubtrees();
    bool is_contained_repeat(const char* sub) const ; 
    void disqualifyContainedRepeats();
    void sortSubtrees(); 
    void enumerateFactors(); 
    void labelFactorSubtrees(); 
    void collectRemainderNodes();  // HMM: Nodes is misleading, in CSGFoundry lingo would be Prim

    void factorize(); 

    int get_num_factor() const ; 
    sfactor& get_factor_(unsigned idx) ; 
    const sfactor& get_factor(unsigned idx) const ; 

    int      get_factor_subtree(unsigned idx) const ; 
    int      get_factor_olvid(unsigned idx) const ; 

    int      get_remainder_subtree() const ; 
    int      get_remainder_olvid() const  ; 

    int      get_ridx_subtree(unsigned ridx) const ; 
    int      get_ridx_olvid(  unsigned ridx) const ; 

    int      get_num_ridx() const ;  

    void get_factor_nodes(std::vector<int>& nodes, unsigned idx) const ; 
    std::string desc_factor() const ; 

    static bool SelectNode( const snode& nd, int q_repeat_index, int q_repeat_ordinal ); 
    // q_repeat_ordinal:-2 selects all repeat_ordinal

    void get_repeat_field(std::vector<int>& result, char q_field , int q_repeat_index, int q_repeat_ordinal ) const ; 
    void get_repeat_lvid( std::vector<int>& lvids, int q_repeat_index, int q_repeat_ordinal=-2 ) const ; 
    void get_repeat_nidx( std::vector<int>& nidxs, int q_repeat_index, int q_repeat_ordinal=-2 ) const ; 

    void get_remainder_nidx(std::vector<int>& nidxs ) const ; 
 
    void get_repeat_node( std::vector<snode>& nodes, int q_repeat_index, int q_repeat_ordinal ) const ; 

    std::string desc_repeat_nodes() const ;  


    void add_inst( glm::tmat4x4<double>& m2w, glm::tmat4x4<double>& w2m, int gas_idx, int nidx ); 
    void add_inst(); 
    void narrow_inst(); 
    void clear_inst(); 
    std::string desc_inst() const ;

    void get_mtindex_range(int& mn, int& mx ) const ; 
    std::string desc_mt() const ; 
    std::string desc_bd() const ; 

    void initStandard() ; 

    int add_material( const char* name, unsigned g4index ); 
    int num_material() const ; 

    int add_surface( const char* name ); 
    int get_surface( const char* name ) const ; 
    int num_surface() const ;   // total including implicit  

    int add_surface( const std::vector<std::string>& names  ); 

    int add_surface_implicit( const char* name ); 
    int get_surface_implicit( const char* name ) const ; 
    int num_surface_implicit() const ; 
    int num_surface_standard() const ; // total with implicit subtracted


    int add_boundary( const int4& bd_ ); 

    const char* get_material_name(int idx) const ; 
    const char* get_surface_name(int idx) const ; 
    std::string get_boundary_name( const int4& bd_, char delim ) const ; 

    NPFold* get_surface_subfold(int idx) const ; 

    void import_bnd(const NP* bnd); 
    void init_mtindex_to_mtline(); 
    int lookup_mtline( int mtindex ) const ; 
};


/**
stree::stree
--------------

Q: why empty NPFold material and surface instead of nullptr ?


     wavelength(nullptr),
     energy(nullptr),
     rayleigh(nullptr),
     mat(nullptr),
     sur(nullptr),
     bd(nullptr),
     bnd(nullptr),
     optical(nullptr)
**/


inline stree::stree()
    :
    level(ssys::getenvint("stree_level", 0)),
    sensor_count(0),
    subs_freq(new sfreq),
#ifdef WITH_SND
    csg(new scsg),
#else
    _csg(new s_csg),
#endif
    standard(new sstandard),
    material(new NPFold),
    surface(new NPFold)
{
    init(); 
}

inline void stree::init()
{
    if(level > 0) std::cout << "stree::init " << std::endl ; 
}


inline void stree::set_level(int level_)
{
    level = level_ ; 
    if(level > 0)
    {
        std::cout 
            << "stree::set_level " << level  
            << " [adjust via envvar SSim__stree_level ]"
            << std::endl 
            ;
    } 
}

inline std::string stree::desc_size(char div) const
{
    std::stringstream ss ;
    ss 
       << std::endl
       << "[stree::desc_size" << div
       << " mtname " << mtname.size() << div 
       << " mtname_no_rindex " << mtname_no_rindex.size() << div 
       << " mtindex " << mtindex.size() << div 
       << " mtline " << mtline.size() << div 
       << " mtindex_to_mtline " << mtindex_to_mtline.size() << div 
       << " suname " << suname.size() << div 
       << " suindex " << suindex.size() << div 
       << " vbd " << vbd.size() << div 
       << " bdname " << bdname.size() << div 
       << " implicit " << implicit.size() << div 
       << " soname " << soname.size() << div 
       << " solids " << solids.size() << div 
       << " sensor_count " << sensor_count << div
       << " nds " << nds.size() << div
       << " rem " << rem.size() << div 
       << " m2w " << m2w.size() << div
       << " w2m " << w2m.size() << div
       << " gtd " << gtd.size() << div
       << " digs " << digs.size() << div
       << " subs " << subs.size() << div
       << " soname " << soname.size() << div
       << " factor " << factor.size() << div
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}


inline std::string stree::desc() const
{
    std::stringstream ss ;
    ss 
       << std::endl
       << "[stree::desc"
       << " level " << level 
       << desc_size() 
       << " stree.desc.subs_freq " 
       << std::endl
       << ( subs_freq ? subs_freq->desc() : "-" )
       << std::endl
       << desc_factor()
       << std::endl
       << desc_repeat_nodes() 
       << std::endl
       ; 

    if(level > 2) ss 
       << "stree::desc.material " 
       << std::endl
       << ( material ? material->desc() : "-" )
       << std::endl
       << " stree::desc.surface "  
       << std::endl
       << ( surface ? surface->desc() : "-" )
       << std::endl
       << " stree::desc.csg "  
       << std::endl
#ifdef WITH_SND
       << ( csg ? csg->desc() : "-" )
#else
       << ( _csg ? _csg->desc() : "-" )
#endif
       << std::endl
       << "]stree::desc"
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}


/**
stree::desc_vec
-----------------

Description of subs_freq showing all subs and frequencies without any cut. 

**/

inline std::string stree::desc_vec() const 
{
    const sfreq::VSU& vsu = subs_freq->vsu ; 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < vsu.size() ; i++)
    {
        const sfreq::SU& su = vsu[i] ; 
        const char* sub = su.first.c_str() ; 
        int freq = su.second ; 

        ss << std::setw(3) << i 
           << " : " 
           << std::setw(32) << sub 
           << " : " 
           << std::setw(6) << freq 
           << std::endl 
           ;
    }
    std::string s = ss.str();     
    return s; 
}

/**
stree::desc_sub
----------------

Description of subtree digests and their frequencies obtained from "(sfreq*)subs_freq" 
For all:false only qualified repeats are listed. 

The depth range of all nodes with each subtree digest is listed
as well as the nidx of the first node with the digest and the 
soname corresponding to that first nidx.::

    st.desc_sub(false)
        0 : 1af760275cafe9ea890bfa01b0acb1d1 : 25600 de:( 6  6) 1st:194249 PMT_3inch_pmt_solid0x66e59f0
        1 : 1e410142530e54d54db8aaaccb63b834 : 12612 de:( 6  6) 1st: 70965 NNVTMCPPMTsMask_virtual0x5f5f900
        2 : 0077df3ebff8aeec56c8a21518e3c887 :  5000 de:( 6  6) 1st: 70972 HamamatsuR12860sMask_virtual0x5f50d40
        3 : 019f9eccb5cf94cce23ff7501c807475 :  2400 de:( 4  4) 1st:322253 mask_PMT_20inch_vetosMask_virtual0x5f62e40
        4 : c051c1bb98b71ccb15b0cf9c67d143ee :   590 de:( 6  6) 1st: 68493 sStrutBallhead0x5853640
        5 : 5e01938acb3e0df0543697fc023bffb1 :   590 de:( 6  6) 1st: 69083 uni10x5832ff0
        6 : cdc824bf721df654130ed7447fb878ac :   590 de:( 6  6) 1st: 69673 base_steel0x58d3270
        7 : 3fd85f9ee7ca8882c8caa747d0eef0b3 :   590 de:( 6  6) 1st: 70263 uni_acrylic10x597c090
        8 : 7d9a644fae10bdc1899c0765077e7a33 :   504 de:( 7  7) 1st:    15 sPanel0x71a8d90

25600+12612+5000+2400+590*4+504+1=48477 this matches the number of GGeo->CSGFoundry inst 

**/

inline std::string stree::desc_sub(bool all) const
{
    unsigned num = subs_freq->get_num();
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = subs_freq->get_key(i);
        int freq = subs_freq->get_freq(i);   // -ve freq means disqualified 
        if(all == false && freq < FREQ_CUT) continue ;
        ss << desc_sub(sub) << std::endl ;  
    }
    std::string s = ss.str();
    return s ;
}

inline std::string stree::desc_sub(const char* sub) const 
{
    int first_nidx = get_first(sub); 

    unsigned mn, mx ; 
    get_depth_range(mn,mx,sub);

    std::stringstream ss ; 
    ss << subs_freq->desc(sub) 
       << " de:"
       << "(" << std::setw(2) << mn
       << " " << std::setw(2) << mx
       << ")"
       << " 1st:" << std::setw(6) << first_nidx
       << " " <<  get_soname(first_nidx)
       ;
    std::string s = ss.str();
    return s ; 
}



/**
stree::Digest
----------------------

Progeny digest needs to include transforms + lvid of subnodes, 
but only lvid of the upper subtree node.  

**/

inline std::string stree::Digest(int lvid, const glm::tmat4x4<double>& tr ) // static
{
    sdigest u ;
    u.add( lvid );
    u.add( (char*)glm::value_ptr(tr), sizeof(double)*16 ) ;
    std::string dig = u.finalize();
    return dig ;
}

inline std::string stree::Digest(int lvid ) // static
{
    return sdigest::Int(lvid);
}


template<typename T>
inline int stree::GetPointerIndex( const std::vector<const T*>& vec, const T* obj) // static
{
    if( obj == nullptr || vec.size() == 0 ) return -1 ; 
    size_t idx = std::distance( vec.begin(), std::find(vec.begin(), vec.end(), obj )); 
    return idx < vec.size() ? int(idx) : -1 ;   
}

template<typename T>
inline int stree::GetValueIndex( const std::vector<T>& vec, const T& obj) // static 
{
    size_t idx = std::distance( vec.begin(), std::find(vec.begin(), vec.end(), obj )); 
    return idx < vec.size() ? int(idx) : -1 ;   
}



inline void stree::get_children( std::vector<int>& children , int nidx ) const
{
    const snode& nd = nds[nidx];
    assert( nd.index == nidx );

    int ch = nd.first_child ;
    while( ch > -1 )
    {
        const snode& child = nds[ch] ;
        assert( child.parent == nd.index );
        children.push_back(child.index);
        ch = child.next_sibling ;
    }
    assert( int(children.size()) == nd.num_child );
}

inline void stree::get_progeny( std::vector<int>& progeny , int nidx ) const
{
    std::vector<int> children ;
    get_children(children, nidx);
    std::copy(children.begin(), children.end(), std::back_inserter(progeny));
    for(unsigned i=0 ; i < children.size() ; i++) get_progeny(progeny, children[i] );
}


inline std::string stree::desc_progeny(int nidx) const 
{
    std::vector<int> progeny ;
    get_progeny(progeny, nidx ); 
    sfreq* sf = make_freq(progeny); 
    sf->sort(); 

    std::stringstream ss ; 
    ss << "stree::desc_progeny nidx " << nidx << " progeny.size " << progeny.size() << std::endl ; 
    ss << "sf.desc" << std::endl << sf->desc() << std::endl ; 
    ss 
       << " i " << std::setw(6) << -1
       << desc_node(nidx, true ) 
       << std::endl
       ; 

    for(unsigned i=0 ; i < progeny.size() ; i++)
    {
        int nix = progeny[i] ;  
        int depth = get_depth(nix);
        ss 
            << " i " << std::setw(6) << i 
            << " depth " << std::setw(2) << depth 
            << desc_node_(nix, sf ) 
            << std::endl
            ; 
    }

    std::string s = ss.str(); 
    return s; 
}


inline void stree::traverse(int nidx) const 
{
    traverse_r(nidx, 0, -1); 
} 

inline void stree::traverse_r(int nidx, int depth, int sibdex) const 
{
    std::vector<int> children ;
    get_children(children, nidx);

    const snode& nd = nds[nidx] ; 

    assert( nd.index == nidx ); 
    assert( nd.depth == depth ); 
    assert( nd.sibdex == sibdex ); 
    assert( nd.num_child == int(children.size()) ); 

    const char* so = get_soname(nidx); 

    if(nd.sensor_id > -1 )
    if(level > 0) std::cout 
        << "stree::traverse_r"
        << " " 
        << nd.desc() 
        << " so " << so 
        << std::endl 
        ; 

    for(unsigned i=0 ; i < children.size() ; i++) traverse_r(children[i], depth+1, i );
}


/**
stree::reorderSensors : changes nd.sensor_index across entire tree
----------------------------------------------------------------------

Invoked from U4Tree::identifySensitive

1. recursive node traverse changing nd.sensor_index 
   into preorder traverse order

2. node loop collecting nd.sensor_id and updating 
   stree::sensor_id vector


This attempts to mimic the preorder traverse sensor order 
used by GGeo/CSG_GGeo to facilitate comparison. 

HMM: I expect the same thing could be done by simply iterating over nds
as the snode are collected in preorder ? 

**/

inline void stree::reorderSensors()
{
    if(level > 0) std::cout 
        << "[ stree::reorderSensors" 
        << std::endl
        ; 

    sensor_count = 0 ; 
    reorderSensors_r(0); 

    if(level > 0) std::cout 
        << "] stree::reorderSensors"
        << " sensor_count " << sensor_count 
        << std::endl 
        ; 

    // change sensor_id vector by looping over 
    // all nodes collecting it when > -1 
    get_sensor_id(sensor_id); 

    assert( sensor_count == sensor_id.size() ); 
}

/**
stree::reorderSensors_r
------------------------

For nodes with sensor_id > -1 change the sensor_index 
into a 0-based preorder traversal count index. 

**/

inline void stree::reorderSensors_r(int nidx)
{
    snode& nd = nds[nidx] ; 

    if( nd.sensor_id > -1 )
    {
        nd.sensor_index = sensor_count ; 
        sensor_count += 1 ; 
    }

    std::vector<int> children ;
    get_children(children, nidx);
    for(unsigned i=0 ; i < children.size() ; i++) reorderSensors_r(children[i]);
}


/**
stree::get_sensor_id from snode nds
-------------------------------------

List *nd.sensor_id* obtained by iterating over all *nds* of the geometry.
As the *nds* vector is in preorder traversal order, the order of 
the *sensor_id* should correspond to *sensor_index* from 0 to num_sensor-1. 

**/

inline void stree::get_sensor_id( std::vector<int>& arg_sensor_id ) const 
{
    arg_sensor_id.clear(); 
    for(unsigned nidx=0 ; nidx < nds.size() ; nidx++)
    {
        const snode& nd = nds[nidx] ; 
        if( nd.sensor_id > -1 ) arg_sensor_id.push_back(nd.sensor_id) ; 
    }
}

/**
stree::postcreate
------------------

Called for reporting from U4Tree::Create 

**/

inline void stree::postcreate() const 
{
    std::cout << "[stree::postcreate" << std::endl ;  

    std::cout << desc_sensor() ; 
    std::cout << desc_sensor_nd(0) ; 
    std::cout << desc_sensor_id(10) ; 

    std::cout << "]stree::postcreate" << std::endl ;  
}

inline std::string stree::desc_sensor() const 
{
    std::stringstream ss ; 
    ss << "stree::desc_sensor" << std::endl 
       << " sensor_id.size " << sensor_id.size() << std::endl 
       << " sensor_count " << sensor_count << std::endl 
       << " sensor_name.size " << sensor_name.size() << std::endl 
       ; 
     
    ss << "sensor_name[" << std::endl  ; 
    for(int i=0; i < int(sensor_name.size()) ; i++) ss << sensor_name[i].c_str() << std::endl ; 
    ss << "]" << std::endl  ; 
    std::string str = ss.str(); 
    return str ;  
}

inline int stree::get_num_nd_sensor() const 
{
    int num_nd = nds.size() ; 
    int num_nd_sensor = 0 ; 
    for(int nidx=0 ; nidx < num_nd ; nidx++) if(nds[nidx].sensor_id > -1) num_nd_sensor += 1 ; 
    return num_nd_sensor ; 
}

inline void stree::get_sensor_nidx( std::vector<int>& sensor_nidx ) const 
{
    int num_nd = nds.size() ; 
    for(int nidx=0 ; nidx < num_nd ; nidx++) 
        if(nds[nidx].sensor_id > -1 ) 
            sensor_nidx.push_back(nidx) ; 
}


inline std::string stree::desc_sensor_nd(int edge) const 
{
    int num_nd = nds.size() ; 
    int num_nd_sensor = get_num_nd_sensor() ; 

    std::vector<int> sensor_nidx ; 
    get_sensor_nidx(sensor_nidx); 

    int num_sid = sensor_nidx.size() ; 
    assert( num_sid == num_nd_sensor ); 

    std::stringstream ss ; 
    ss << "[stree::desc_sensor_nd" << std::endl ; 
    ss << " edge            " << edge << std::endl ; 
    ss << " num_nd          " << num_nd << std::endl ; 
    ss << " num_nd_sensor   " << num_nd_sensor << std::endl ; 
    ss << " num_sid         " << num_sid << std::endl ; 

    int offset = -1 ;  

    for(int i=0 ; i < num_sid ; i++)
    {
        int nidx = sensor_nidx[i] ; 
        int n_nidx = i < num_sid - 1 ? sensor_nidx[i+1] : sensor_nidx[i] ;  

        const snode& nd = nds[nidx] ;
        const snode& n_nd = nds[n_nidx] ; 

        int sid = nd.sensor_id ;  
        int n_sid = n_nd.sensor_id ;  

        assert( sid > -1 ); 
        assert( n_sid > -1 ); 

        bool head = i < edge ; 
        bool tail = (i > (num_sid - edge)) ;  
        bool tran = std::abs(n_sid - sid) > 1 ; 

        if(tran) offset=0 ; 
        bool tran_post = offset > -1 && offset < 4 ; 

        if(head || tail || tran || tran_post) 
        {
            ss 
                << " nidx " << std::setw(6) << nidx     
                << " i " << std::setw(6) << i     
                << " sensor_id " << std::setw(6) << nd.sensor_id
                << " sensor_index " << std::setw(6) << nd.sensor_index
                << " sensor_name " << std::setw(6) << nd.sensor_name
                << std::endl
                ;
        }
        else if(i == edge) 
        {
            ss << "..." << std::endl ; 
        }
        offset += 1 ; 
    }
    ss << "]stree::desc_sensor_nd" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}



inline std::string stree::desc_sensor_id(unsigned edge) const 
{
    unsigned num_sid = sensor_id.size() ; 
    std::stringstream ss ; 
    ss << "stree::desc_sensor_id sensor_id.size " 
       << num_sid 
       << std::endl 
       << "[" 
       << std::endl 
       ; 

    int offset = -1 ;  
    for(unsigned i=0 ; i < num_sid ; i++)
    {  
        int sid = sensor_id[i] ;  
        int nid = i < num_sid - 1 ? sensor_id[i+1] : sid ;  

        bool head = i < edge ; 
        bool tail = (i > (num_sid - edge)) ;  
        bool tran = std::abs(nid - sid) > 1 ; 

        if(tran) offset=0 ; 
        bool tran_post = offset > -1 && offset < 4 ; 

        if(head || tail || tran || tran_post) 
        {
            ss << std::setw(7) << i << " sid " << std::setw(8) << sid << std::endl ; 
        }
        else if(i == edge) 
        {
            ss << "..." << std::endl ; 
        }
        offset += 1 ; 
    }   
    ss << "]" ; 
    std::string s = ss.str(); 
    return s ; 
}

inline std::string stree::DescSensor( const std::vector<int>& sensor_identifier, const std::vector<int>& sensor_index, unsigned edge )  // static
{
    assert( sensor_identifier.size() == sensor_index.size() ); 
    unsigned num_sensor = sensor_identifier.size() ; 

    std::stringstream ss ; 
    ss << "stree::DescSensor num_sensor " << num_sensor << std::endl ;   
    unsigned offset = 0 ; 
    for(unsigned i=0 ; i < num_sensor ; i++)
    {
        int s_index      = sensor_index[i] ;  
        int s_identifier = sensor_identifier[i] ;  
        int n_identifier = i < num_sensor - 1 ? sensor_identifier[i+1] : s_identifier ; 

        bool jump = std::abs(n_identifier - s_identifier) > 1 ; // sensor identifier transition
        if(jump) offset = 0 ; 

        if( i < edge || i > num_sensor - edge || offset < 5 ) 
        {
            ss
                << " i " << std::setw(7) << i 
                << " s_index " << std::setw(7) << s_index
                << " s_identifier " << std::setw(7) << s_identifier
                << std::endl 
                ;
        }
        else if( i == edge || i == num_sensor - edge ) 
        {
            ss << "..." << std::endl ; 
        }
        offset += 1 ; 
    }

    std::string s = ss.str(); 
    return s ; 
}





/**
stree::lookup_sensor_identifier
---------------------------------

1. arg_sensor_identifier vector is resized to match the size of arg_sensor_index 
2. for arg_sensor_index that are not "not-a-sensor" returns s_identifier from stree::sensor_id vector

This is used from CSG_GGeo_Convert::addInstances

**/

inline void stree::lookup_sensor_identifier( 
             std::vector<int>& arg_sensor_identifier, 
       const std::vector<int>& arg_sensor_index, 
       bool one_based_index, 
       bool verbose, 
       unsigned edge ) const 
{
    if(verbose) std::cerr 
         << "stree::lookup_sensor_identifier.0"
         << " arg_sensor_identifier.size " << arg_sensor_identifier.size() 
         << " arg_sensor_index.size " << arg_sensor_index.size() 
         << " sensor_id.size " << sensor_id.size() 
         << " edge " << edge 
         << std::endl 
         ;

    arg_sensor_identifier.resize(arg_sensor_index.size()); 

    unsigned num_lookup = arg_sensor_index.size() ; 
    for(unsigned i=0 ; i < num_lookup ; i++)
    {   
        int s_index = one_based_index ? arg_sensor_index[i] - 1 : arg_sensor_index[i] ; 
        // "correct" 1-based to be 0-based 
        bool s_index_inrange = s_index > -1 && s_index < int(sensor_id.size()) ; 
        int s_identifier = s_index_inrange ? sensor_id[s_index] : -1 ; 
        arg_sensor_identifier[i] = s_identifier ; 

        if(verbose)
        {
            if( i < edge || i > num_lookup - edge) 
            {
                std::cerr 
                    << "stree::lookup_sensor_identifier.1"
                    << " i " << std::setw(3) << i  
                    << " s_index " << std::setw(7) << s_index 
                    << " s_index_inrange " << std::setw(1) << s_index_inrange 
                    << " s_identifier " << std::setw(7) << s_identifier 
                    << " sensor_id.size " << std::setw(7) << sensor_id.size()
                    << std::endl 
                    ;  
            }
            else if( i == edge )
            {
                std::cerr 
                    << "stree::lookup_sensor_identifier.1"
                    << " i " << std::setw(3) << i  
                    << " ... "
                    << std::endl 
                    ;
            }
        }      // verbose
    }          // over num_lookup
} 


inline sfreq* stree::make_progeny_freq(int nidx) const 
{
    std::vector<int> progeny ; 
    get_progeny(progeny, nidx ); 
    return make_freq(progeny);  
}

inline sfreq* stree::make_freq(const std::vector<int>& nodes ) const
{
    sfreq* sf = new sfreq ; 
    for(unsigned i=0 ; i < nodes.size() ; i++)
    {
        int nidx = nodes[i]; 
        sf->add(get_sub(nidx)); 
    }
    return sf ; 
} 

inline int stree::find_lvid(const char* q_soname, bool starting ) const 
{
    int lvid = -1 ; 
    for(unsigned i=0 ; i < soname.size() ; i++)
    {
        const char* name = soname[i].c_str(); 
        if(sstr::Match( name, q_soname, starting))
        {
            lvid = i ; 
            break ; 
        }  
    }
    return lvid ; 
}

inline void stree::find_lvid_nodes_( std::vector<snode>& nodes, int lvid ) const 
{
    for(unsigned i=0 ; i < nds.size() ; i++)
    {
        const snode& sn = nds[i] ; 
        assert( int(i) == sn.index ); 
        if(sn.lvid == lvid) nodes.push_back(sn) ; 
    }
}

inline void stree::find_lvid_nodes( std::vector<int>& nodes, int lvid ) const 
{
    for(unsigned i=0 ; i < nds.size() ; i++)
    {
        const snode& sn = nds[i] ; 
        assert( int(i) == sn.index ); 
        if(sn.lvid == lvid) nodes.push_back(sn.index) ; 
    }
}

inline void stree::find_lvid_nodes( std::vector<int>& nodes, const char* q_soname, bool starting ) const 
{
    int lvid = find_lvid(q_soname, starting); 
    find_lvid_nodes(nodes, lvid ); 
}

inline int stree::find_lvid_node( const char* q_soname, int ordinal ) const 
{
    std::vector<int> nodes ; 
    find_lvid_nodes(nodes, q_soname ); 
    if(ordinal < 0) ordinal += nodes.size() ; // -ve ordinal counts from back 

    return ordinal > -1 && ordinal < int(nodes.size()) ? nodes[ordinal] : -1 ; 
}

/**
stree::find_lvid_node
----------------------



**/
inline int stree::find_lvid_node( const char* q_spec ) const 
{
    std::vector<std::string> elem ; 
    sstr::Split(q_spec, ':', elem );  

    const char* q_soname  = elem.size() > 0 ? elem[0].c_str() : nullptr ; 
    const char* q_middle  = elem.size() > 1 ? elem[1].c_str() : nullptr ; 
    const char* q_ordinal = elem.size() > 2 ? elem[2].c_str() : nullptr ; 

    int middle  = q_middle  ? std::atoi(q_middle)  : 0 ; 
    int ordinal = q_ordinal ? std::atoi(q_ordinal) : 0 ; 

    assert( middle == 0 ); // slated for use with global addressing (like MOI)

    int nidx = find_lvid_node(q_soname, ordinal); 
    return nidx ; 
}

// TODO: should this be using sfactor ?
inline void stree::get_sub_sonames( std::vector<std::string>& sonames ) const 
{
    std::vector<std::string> subs ; 
    subs_freq->get_keys(subs, FREQ_CUT ); 
    for(unsigned i=0 ; i < subs.size() ; i++)
    {
        const char* sub = subs[i].c_str(); 
        const char* soname_ = get_sub_soname(sub); 
        sonames.push_back(soname_);  
    }
}


inline const char* stree::get_sub_soname(const char* sub) const 
{
    int first_nidx = get_first(sub); 
    return first_nidx == -1 ? nullptr : get_soname(first_nidx ) ; 
}



inline std::string stree::Name( const std::string& name, bool strip ) // static
{
    return strip ? sstr::StripTail(name, "0x") : name ; 
}
inline std::string stree::get_lvid_soname(int lvid, bool strip) const 
{
    if(lvid < 0 || lvid >= int(soname.size())) return "bad_lvid" ;  
    return Name(soname[lvid], strip) ; 
}

inline void stree::get_meshname( std::vector<std::string>& names) const 
{
    assert( names.size() == 0 ); 
    for(unsigned i=0 ; i < soname.size() ; i++) names.push_back( Name(soname[i],true) ); 
}

inline void stree::get_mmlabel( std::vector<std::string>& names) const 
{
    assert( names.size() == 0 ); 
    int num_ridx = get_num_ridx(); 
    for(int ridx=0 ; ridx < num_ridx ; ridx++)
    {
        int num_prim = get_ridx_subtree(ridx) ; 
        int olvid    = get_ridx_olvid(ridx) ; 

        assert( olvid < int(soname.size()) ); 
        std::string name = get_lvid_soname(olvid, true); 

        std::stringstream ss ;  
        ss << num_prim << ":" << name ; 
        std::string str = ss.str(); 
        names.push_back(str);  
    }
}


inline int stree::get_num_nodes() const 
{
    return nds.size() ; 
}

inline const char* stree::get_soname(int nidx) const
{
    return nidx > -1 ? soname[nds[nidx].lvid].c_str() : "?" ;
}
inline const char* stree::get_sub(int nidx) const 
{
    return nidx > -1 ? subs[nidx].c_str() : nullptr ; 
}
inline int stree::get_depth(int nidx) const 
{ 
    return nidx > -1 ? nds[nidx].depth : -1 ; 
}
inline int stree::get_parent(int nidx) const 
{ 
    return nidx > -1 ? nds[nidx].parent : -1 ; 
}
inline int stree::get_lvid(int nidx) const 
{ 
    return nidx > -1 ? nds[nidx].lvid : -1 ; 
}
inline int stree::get_copyno(int nidx) const 
{ 
    return nidx > -1 ? nds[nidx].copyno : -1 ; 
}

inline const snode* stree::get_node(int nidx) const 
{
    return &nds[nidx] ; 
}
inline const snode* stree::get_parent_node(int nidx) const 
{
    const snode* n0 = &nds[nidx] ; 
    const snode* nd = n0->parent > -1 ? &nds[n0->parent] : nullptr ; 
    return nd ; 
}

/**
stree::is_outer_node
----------------------

An outer node is either the root node with no parent
or has a parent with a different repeat_index.
The outer nodes correspond to base nodes of the instances. 

**/

inline bool stree::is_outer_node(int nidx) const 
{
    const snode* n = get_node(nidx); 
    assert(n); 
    const snode* p = get_parent_node(nidx); 

    return p == nullptr ? 
                            true  
                         :
                            p->repeat_index != n->repeat_index 
                         ;

}



/**
stree::get_ancestors
---------------------

This should work even during node collection immediately 
after the parent link has been set and the snode pushed back. 

local:false
    Collects parent, then parent-of-parent and so on 
    until reaching root (nidx:0) which has no parent.  
    Then reverses the list to put into root first order. 

    At first glance you might think this would miss root, but that 
    is not the case as it collects parents and the node prior 
    to the parent results in collecting root nidx:0. 

local:true
    Gets ancestors of *nidx* that have the same repeat_index as the *nidx* node.
    For *nidx* within the remainder nodes this is expected to start from root, nidx 0.
    For *nidx* within instanced nodes this will only include nodes within that same instance. 



Q: Judgement after collection, so does that correctly skip the outer ?  
A: No because its recording the parent looking ahead, hence try popping the last 

**/

inline void stree::get_ancestors( std::vector<int>& ancestors, int nidx, bool local, std::ostream* out ) const
{
    const snode* nd0 = &nds[nidx] ; 
    const snode* nd = nd0->parent > -1 ? &nds[nd0->parent] : nullptr ; 

    while(nd)
    {
        if(local == false)
        {
            ancestors.push_back(nd->index);
        }
        else if( !is_outer_node(nd->index) && nd->repeat_index == nd0->repeat_index )
        {
            ancestors.push_back(nd->index);
        }

        nd = nd->parent > -1 ? &nds[nd->parent] : nullptr ; 
    }
    std::reverse( ancestors.begin(), ancestors.end() );

    if(out) 
    {
        int num_ancestors = ancestors.size() ; 
        *out 
            << "stree::get_ancestors"
            << " nidx " << nidx 
            << " local " << local
            << " nd0.repeat_index " << ( nd0 ? nd0->repeat_index : -10 ) 
            << " nd.repeat_index "  << ( nd  ? nd->repeat_index  : -10 )
            ;

        *out 
            << " num_ancestors " << num_ancestors
            << " [" 
            ; 
        for(int i=0 ; i < num_ancestors ; i++) *out << ancestors[i] << " " ; 
        *out << "]" << std::endl ; 

        bool show_sub = true ; 
        for(int i=0 ; i < num_ancestors ; i++) *out << desc_node(ancestors[i], show_sub) << std::endl ;
        *out << desc_node(nidx, show_sub) << " " << std::endl ; 
    }


}


inline std::string stree::desc_ancestors(int nidx, bool local) const
{
    std::stringstream ss ; 
    std::vector<int> ancestors ;
    get_ancestors(ancestors, nidx, local, &ss );

    ss << "stree::desc_ancestors" << std::endl ; 
    std::string str = ss.str(); 
    ss << str ; 
    std::string s = ss.str();
    return s ; 
}






inline void stree::get_node_transform( glm::tmat4x4<double>& m2w_, glm::tmat4x4<double>& w2m_, int nidx ) const 
{
    assert( w2m.size() == m2w.size() ); 
    assert( nidx > -1 && nidx < int(m2w.size())); 

    m2w_ = m2w[nidx]; 
    w2m_ = w2m[nidx]; 
}

inline void stree::get_node_product( 
                      glm::tmat4x4<double>& m2w_, 
                      glm::tmat4x4<double>& w2m_, int nidx, bool local, bool reverse, std::ostream* out ) const 
{
    std::vector<int> nodes ; 
    get_ancestors(nodes, nidx, local, out);  // root-first-order (from collecting parent links then reversing the vector)

    bool is_local_outer = local && is_outer_node(nidx) ; 
    if(is_local_outer == false ) nodes.push_back(nidx); 

    int num_nodes = nodes.size();

    if(out)
    {
        *out << "stree::get_node_product"  
             << " nidx " << nidx
             << " local " << local 
             << " reverse " << reverse 
             << " is_local_outer " << is_local_outer
             << " num_nodes " << num_nodes 
             << " [" 
             ; 
        for(int i=0 ; i < num_nodes ; i++ ) *out << " " << nodes[i] ; 
        *out << "]" << std::endl ; 
    }


    glm::tmat4x4<double> tp(1.); 
    glm::tmat4x4<double> vp(1.); 

    for(int i=0 ; i < num_nodes ; i++ ) 
    {
        int j = num_nodes - 1 - i ;  
        int ii = nodes[reverse ? j : i] ; 
        int jj = nodes[reverse ? i : j] ; 

        if(out)
        { 
            const char* s_ii = get_soname(ii) ; 
            const char* s_jj = get_soname(jj) ; 

            *out 
               << std::endl 
               << " i " << i
               << " j " << j
               << " ii " << ii
               << " jj " << jj
               << " s_ii " << s_ii
               << " s_jj " << s_jj
               << std::endl 
               ;
        }

        glm::tmat4x4<double> it(1.); 
        glm::tmat4x4<double> iv(1.); 
        get_node_transform( it, iv, ii ); 
        if(out) *out << stra<double>::Desc(it, iv, "it", "iv" ); 

        glm::tmat4x4<double> jt(1.); 
        glm::tmat4x4<double> jv(1.); 
        get_node_transform( jt, jv, jj ); 
        if(out) *out << stra<double>::Desc(jt, jv, "jt", "jv" ); 

        tp *= it ; 
        vp *= jv ; // inverse-transform product in opposite order

        //if(out) *out << stra<double>::Desc(tp, vp, "tp", "vp" );   // product not always identity 
    }

    if(out) *out << stra<double>::Desc(tp, vp, "tp", "vp" ); 

    memcpy( glm::value_ptr(m2w_), glm::value_ptr(tp), sizeof(glm::tmat4x4<double>) );
    memcpy( glm::value_ptr(w2m_), glm::value_ptr(vp), sizeof(glm::tmat4x4<double>) );
}


inline std::string stree::desc_node_product( glm::tmat4x4<double>& m2w_, glm::tmat4x4<double>& w2m_, int nidx, bool local, bool reverse ) const 
{
    std::stringstream ss ; 
    ss << "stree::desc_node_product" ; 
    get_node_product( m2w_, w2m_, nidx, local, reverse, &ss ); 
    std::string s = ss.str(); 
    return s ; 
}

/**
stree::get_combined_transform : combining structural and CSG transforms
------------------------------------------------------------------------

* combines structural (volume level) and CSG (solid level) transforms
* canonical usage from CSGImport::importNode

The CSG constituent *snd/sn* lvid is required to directly match that of 
the structural *snode*, not just by containment but directly.  


modelFrame
   typically close to origin coordinates

worldFrame
   typically far from origin coordinates

m2w
   modelFrame to worldFrame
w2m 
   worldFrame to modelFrame 

get_m2w_product 
    product of m2w transforms from root then down volume tree to the structural snode *index*

get_w2m_product 
    product of w2m transforms from snode *index* then up the structural tree


GParts::applyPlacementTransform does::

    1243     for(unsigned i=0 ; i < ni ; i++)
    1244     {
    1245         nmat4triple* tvq = m_tran_buffer->getMat4TriplePtr(i) ;
    1247         bool match = true ;
    1248         const nmat4triple* ntvq = nmat4triple::make_transformed( tvq, placement, reversed, "GParts::applyPlacementTransform", match );
    1251         if(!match) mismatch.push_back(i);
    1253         m_tran_buffer->setMat4Triple( ntvq, i );
       


sysrap/tests/stree_create_test.cc 
   sets up a geometry with structural translations+rotations 
   and csg translations+rotations+scales to test the get_transform product ordering  

**/

template<typename N>
inline void stree::get_combined_transform(
    glm::tmat4x4<double>& t,
    glm::tmat4x4<double>& v,
    const snode& node,
    const N* nd,
    std::ostream* out ) const
{
    bool local = node.repeat_index > 0 ;   // for instanced nodes restrict to same repeat_index excluding outer 

    glm::tmat4x4<double> tt(1.) ;
    glm::tmat4x4<double> vv(1.) ;

    get_node_product( tt, vv, node.index, local, false, out ); // reverse:false


    glm::tmat4x4<double> tc(1.) ;
    glm::tmat4x4<double> vc(1.) ;

    if(nd)
    {
        assert( node.lvid == nd->lvid );
        N::NodeTransformProduct(nd->idx(), tc, vc, false, out );  // reverse:false
    }

    // combine structural (volume level) and CSG (solid level) transforms
    t = tt * tc ;
    v = vc * vv ;

    if(out) *out << stra<double>::Desc( t, v, "(tt*tc)", "(vc*vv)" ) << std::endl << std::endl ;
}

template<typename N>
inline std::string stree::desc_combined_transform(
    glm::tmat4x4<double>& t,
    glm::tmat4x4<double>& v,
    const snode& node,
    const N* nd ) const
{
    std::stringstream ss ;
    ss << "stree::desc_combined_transform" << std::endl;
    get_combined_transform(t, v, node, nd, &ss );
    std::string str = ss.str();
    return str ;
}

/**
stree::get_nodes : node indices of nodes with *sub* subtree digest
---------------------------------------------------------------------

Collects node indices of all nodes with the subtree digest provided in the argument. 
This simply matches against the subs vector that contains subtree digest 
strings for all nodes in node order. 

Hence when providing the *sub* digest of factor subtrees this 
will return the node indices of all "outer volume" nodes 
of that factor.  This is done by stree::get_factor_nodes

**/

inline void stree::get_nodes(std::vector<int>& nodes, const char* sub) const
{
    for(unsigned i=0 ; i < subs.size() ; i++) 
        if(strcmp(subs[i].c_str(), sub)==0) 
            nodes.push_back(int(i)) ;
}

inline void stree::get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const
{
    std::vector<int> nodes ;
    get_nodes(nodes, sub);
    mn = 100 ;
    mx = 0 ;
    for(unsigned i=0 ; i < nodes.size() ; i++)
    {
        unsigned nidx = nodes[i];
        const snode& sn = nds[nidx] ;
        if( unsigned(sn.depth) > mx ) mx = sn.depth ;
        if( unsigned(sn.depth) < mn ) mn = sn.depth ;
    }
}


inline int stree::get_first( const char* sub ) const
{
    for(unsigned i=0 ; i < subs.size() ; i++) if(strcmp(subs[i].c_str(), sub)==0) return int(i) ;
    return -1 ;
}


inline std::string stree::subtree_digest(int nidx) const
{
    std::vector<int> progeny ;
    get_progeny(progeny, nidx);

    sdigest u ;
    u.add( nds[nidx].lvid );  // just lvid of subtree top, not the transform
    for(unsigned i=0 ; i < progeny.size() ; i++) u.add(digs[progeny[i]]) ;
    return u.finalize() ;
}

inline std::string stree::depth_spacer(int depth) // static
{
    std::string spacer(MAXDEPTH, ' ');  
    if(depth < MAXDEPTH) spacer[depth] = '+' ; 
    return spacer ; 
}

inline std::string stree::desc_node_(int nidx, const sfreq* sf) const
{
    const snode& nd = nds[nidx];
    const char* sub = subs[nidx].c_str();
    assert( nd.index == nidx );
    bool is_outer = is_outer_node(nidx); 

    std::stringstream ss ;
    ss << depth_spacer(nd.depth) ; 
    ss << nd.desc() ;
    ss << " ou " << ( is_outer ? "Y" : "N" ) ; 
    if(sf) ss << " " << sf->desc(sub) ;
    ss << " " << soname[nd.lvid]  ;
    std::string s = ss.str();
    return s ;
}
inline std::string stree::desc_node(int nidx, bool show_sub) const
{
   const sfreq* sf = show_sub ? subs_freq : nullptr ; 
   return desc_node_(nidx, sf );  
}



inline std::string stree::desc_nodes(const std::vector<int>& nn, int edgeitems) const
{
    int num = nn.size(); 
    std::stringstream ss ;
    ss << "stree::desc_nodes " << num << std::endl ;
    for(int i=0 ; i < num ; i++)
    {
        if( i < edgeitems || ( i > num - edgeitems ))
        {
            ss << desc_node(nn[i]) << std::endl ;
        }
        else if( i == edgeitems ) 
        {
            ss << " ... " << std::endl ; 
        }
    }
    std::string s = ss.str();
    return s ;
}

// HMM: could use template specialization to avoid the duplication here
inline std::string stree::desc_nodes_(const std::vector<snode>& nn, int edgeitems) const
{
    int num = nn.size(); 
    std::stringstream ss ;
    ss << "stree::desc_nodes_ " << num << std::endl ;
    for(int i=0 ; i < num ; i++)
    {
        if( i < edgeitems || ( i > num - edgeitems ))
        {
            ss << desc_node(nn[i].index) << std::endl ;
        }
        else if( i == edgeitems ) 
        {
            ss << " ... " << std::endl ; 
        }
    }
    std::string s = ss.str();
    return s ;
}






inline std::string stree::desc_solids() const 
{
    std::stringstream ss ; 
    ss << "stree::desc_solids" << std::endl ; 
    for(int nidx=0 ; nidx < get_num_nodes() ; nidx++) 
    {
        ss
            << " nidx " << std::setw(6) << nidx 
            << " so " << get_soname(nidx)
            << std::endl 
            ;   
    }
    std::string s = ss.str();
    return s ; 
}


/**
stree::make_trs
-----------------

This is used from U4Tree::simtrace_scan as the basis for u4/tests/U4SimtraceTest.sh 

1. HMM: this is based on GTD: "GGeo Transform Debug" so it is not future safe 

   * TODO: adopt the modern equivalent of GTD, or create one if non-existing 

2. saves solid names for every node of the geometry, so thats lots of
   repeated solid names in full geometries 

3. implication is that the number of nodes in the geometry 
   matches the number of gtd and trs transforms (CHECK THAT)

**/

inline NP* stree::make_trs() const
{
    NP* trs = NP::Make<double>( gtd.size(), 4, 4 ); 
    trs->read2<double>( (double*)gtd.data() ) ; 

    std::vector<std::string> nd_soname ; 
    int num_nodes = get_num_nodes(); 
    for(int nidx=0 ; nidx < num_nodes ; nidx++)
    {
        const char* so = get_soname(nidx); 
        nd_soname.push_back(so);    
    }   
    trs->set_names(nd_soname); 

    return trs ; 
}

inline void stree::save_trs(const char* fold) const 
{
    NP* trs = make_trs();  
    trs->save(fold, TRS ); 
}









inline void stree::save( const char* base, const char* reldir ) const 
{
    const char* dir = U::Resolve(base, reldir); 
    save_(dir); 
}
inline void stree::save_( const char* dir ) const 
{
    NPFold* fold = serialize() ; 
    fold->save(dir) ;  
}

inline NPFold* stree::serialize() const 
{
    NPFold* fold = new NPFold ; 

    NP* _nds = NPX::ArrayFromVec<int,snode>( nds, snode::NV ) ; 
    NP* _rem = NPX::ArrayFromVec<int,snode>( rem, snode::NV ) ; 
    NP* _m2w = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( m2w, 4, 4 ) ; 
    NP* _w2m = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( w2m, 4, 4 ) ; 
    NP* _gtd = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( gtd, 4, 4 ) ; 

    fold->add( NDS, _nds ); 
    fold->add( REM, _rem );
    fold->add( M2W, _m2w );
    fold->add( W2M, _w2m );
    fold->add( GTD, _gtd );

    NP* _mtname = NPX::Holder(mtname) ; 
    NP* _mtname_no_rindex = NPX::Holder(mtname_no_rindex) ; 
    NP* _mtindex = NPX::ArrayFromVec<int,int>( mtindex ); 
    NP* _mtline = NPX::ArrayFromVec<int,int>( mtline ); 


    fold->add( MTNAME,  _mtname ); 
    fold->add( MTNAME_NO_RINDEX,  _mtname_no_rindex ); 
    fold->add( MTINDEX, _mtindex ); 
    fold->add( MTLINE , _mtline ); 

    if(material) fold->add_subfold( MATERIAL, material );  
    if(surface)  fold->add_subfold( SURFACE,  surface );  


    fold->add( SUNAME,   NPX::Holder(suname) ); 
    fold->add( IMPLICIT, NPX::Holder(implicit) ); 
    fold->add( SUINDEX,  NPX::ArrayFromVec<int,int>( suindex )  ); 

    NPFold* f_standard = standard->serialize() ; 
    fold->add_subfold( STANDARD, f_standard ); 

#ifdef WITH_SND
    fold->add_subfold( CSG, csg->serialize() ); 
#else
    fold->add_subfold( _CSG, _csg->serialize() ); 
#endif

    fold->add( SONAME, NPX::Holder(soname) ); 

    fold->add( DIGS, NPX::Holder(digs) ); 
    fold->add( SUBS, NPX::Holder(subs) ); 

    NPFold* f_subs_freq = subs_freq->serialize() ; 
    fold->add_subfold( SUBS_FREQ, f_subs_freq ); 

    NP* _factor = NPX::ArrayFromVec<int,sfactor>( factor, sfactor::NV ); 

    NP* _inst = NPX::ArrayFromVec<double, glm::tmat4x4<double>>( inst, 4, 4) ; 
    NP* _iinst = NPX::ArrayFromVec<double, glm::tmat4x4<double>>( iinst, 4, 4) ; 

    // _f4 just for debug comparisons : narrowing normally only done in memory for upload  
    NP* _inst_f4 = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( inst_f4, 4, 4) ; 
    NP* _iinst_f4 = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( iinst_f4, 4, 4) ; 

    NP* _inst_nidx = NPX::ArrayFromVec<int,int>( inst_nidx ) ; 
    NP* _sensor_id = NPX::ArrayFromVec<int,int>( sensor_id ) ; 
    NP* _sensor_name = NPX::Holder(sensor_name); 

    fold->add( FACTOR, _factor ); 
    fold->add( INST,   _inst ); 
    fold->add( IINST,   _iinst ); 
    fold->add( INST_F4,   _inst_f4 ); 
    fold->add( IINST_F4,   _iinst_f4 ); 
    fold->add( INST_NIDX,   _inst_nidx ); 
    fold->add( SENSOR_ID,   _sensor_id ); 
    fold->add( SENSOR_NAME, _sensor_name ); 

    return fold ; 
}




template<typename S, typename T>
inline void stree::ImportArray( std::vector<S>& vec, const NP* a )
{
    if(a == nullptr) return ; 
    vec.resize(a->shape[0]);
    memcpy( (T*)vec.data(),    a->cvalues<T>() ,    a->arr_bytes() );
}

template void stree::ImportArray<snode, int>(std::vector<snode>& , const NP* ); 
template void stree::ImportArray<int  , int>(std::vector<int>&   , const NP* ); 
template void stree::ImportArray<int4 , int>(std::vector<int4>&  , const NP* ); 
template void stree::ImportArray<glm::tmat4x4<double>, double>(std::vector<glm::tmat4x4<double>>& , const NP* ); 
template void stree::ImportArray<glm::tmat4x4<float>, float>(std::vector<glm::tmat4x4<float>>& , const NP* ); 
template void stree::ImportArray<sfactor, int>(std::vector<sfactor>& , const NP* ); 

inline void stree::ImportNames( std::vector<std::string>& names, const NP* a ) // static 
{
    if( a == nullptr )
    {
        std::cerr << "stree::ImportNames a is null " << std::endl ; 
        return ;  
    }
    a->get_names(names); 
}





inline stree* stree::Load(const char* base, const char* reldir ) // static 
{
    stree* st = new stree ; 
    st->load(base, reldir); 
    return st ; 
}
inline int stree::load( const char* base, const char* reldir ) 
{
    const char* dir = U::Resolve(base, reldir ); 
    int rc = load_(dir); 
    return rc ; 
}

inline int stree::load_( const char* dir )
{
    if(level > 0) std::cerr << "stree::load_ " << ( dir ? dir : "-" ) << std::endl ; 
    NPFold* fold = NPFold::Load(dir) ; 
    import(fold); 
    return 0 ; 
}




inline void stree::import(const NPFold* fold)
{
    if( fold == nullptr )
    {
        std::cerr 
            << "stree::import"
            << " : ERROR : null fold " 
            << std::endl ; 
        return ; 
    }

    ImportArray<snode, int>( nds,                  fold->get(NDS) );
    ImportArray<snode, int>( rem,                  fold->get(REM) ); 
    ImportArray<glm::tmat4x4<double>, double>(m2w, fold->get(M2W) ); 
    ImportArray<glm::tmat4x4<double>, double>(w2m, fold->get(W2M) ); 
    ImportArray<glm::tmat4x4<double>, double>(gtd, fold->get(GTD) ); 

    ImportNames( soname,            fold->get(SONAME) ); 
    ImportNames( mtname,            fold->get(MTNAME) );
    ImportNames( mtname_no_rindex,  fold->get(MTNAME_NO_RINDEX) );
    ImportNames( suname,            fold->get(SUNAME) );
    ImportNames( implicit,          fold->get(IMPLICIT) );

    ImportArray<int, int>( mtindex, fold->get(MTINDEX) );
    ImportArray<int, int>( suindex, fold->get(SUINDEX) );
  
    NPFold* f_standard = fold->get_subfold(STANDARD) ; 
    standard->import(f_standard); 

    assert( standard->bd ); 
    NPX::VecFromArray<int4>( vbd, standard->bd );  
    standard->bd->get_names( bdname );

    assert( standard->bnd ); 
    import_bnd( standard->bnd ); 

    /*
    const NP* _mtline = fold->get(MTLINE) ; 
    std::cout 
        << "stree::import"
        << " _mtline " << ( _mtline ? _mtline->sstr() : "-" ) 
        << std::endl
        ; 
 
    ImportArray<int, int>( mtline, _mtline );
    // _mtline is empty so this stomps of the 
    // one done by import_bnd/init_mtindex_to_mtline

    init_mtindex_to_mtline();   
    // NO : THIS IS DONE BY import_bnd
    */


#ifdef WITH_SND
    NPFold* csg_f = fold->get_subfold(CSG) ;
    if(csg_f == nullptr) std::cerr
        << "stree::import.WITH_SND "
        << " FAILED : DUE TO LACK OF subfold CSG : " << CSG 
        << std::endl 
        << " PROBABLY THE PERSISTED TREE WAS CREATED NOT:WITH_SND "
        << std::endl 
        << " SWITCH OPTION TO NOT:WITH_SND OR RECREATE THE TREE WITH_SND "
        << std::endl 
        ;

    assert(csg_f);  
    csg->import(csg_f); 
#else
    NPFold* csg_f = fold->get_subfold(_CSG) ;

    if(csg_f == nullptr) std::cerr
        << "stree::import.NOT:WITH_SND"
        << " FAILED : DUE TO LACK OF subfold _CSG : " << _CSG 
        << std::endl 
        << " PROBABLY THE PERSISTED TREE WAS CREATED WITH_SND "
        << std::endl 
        << " SWITCH OPTION TO WITH_SND OR RECREATE THE TREE NOT:WITH_SND "
        << std::endl 
        ;

    _csg->import(csg_f); 
#endif


    ImportNames( digs, fold->get(DIGS) ); 
    ImportNames( subs, fold->get(SUBS) ); 

    NPFold* f_subs_freq = fold->get_subfold(SUBS_FREQ) ; 
    subs_freq->import(f_subs_freq); 

    ImportArray<sfactor, int>( factor, fold->get(FACTOR) ); 

    material = fold->get_subfold( MATERIAL) ;
    surface  = fold->get_subfold( SURFACE ) ;

    ImportArray<glm::tmat4x4<double>, double>(inst,   fold->get(INST)); 
    ImportArray<glm::tmat4x4<double>, double>(iinst,  fold->get(IINST)); 
    ImportArray<glm::tmat4x4<float>, float>(inst_f4,  fold->get(INST_F4)); 
    ImportArray<glm::tmat4x4<float>, float>(iinst_f4, fold->get(IINST_F4)); 

    ImportArray<int, int>( sensor_id, fold->get(SENSOR_ID) );
    sensor_count = sensor_id.size(); 
    ImportNames( sensor_name, fold->get(SENSOR_NAME) ); 

 
    ImportArray<int, int>( inst_nidx, fold->get(INST_NIDX) );
}



inline int stree::Compare( const std::vector<int>& a, const std::vector<int>& b ) // static 
{
    if( a.size() != b.size() ) return -1 ;
    int mismatch = 0 ;
    for(unsigned i=0 ; i < a.size() ; i++) if(a[i] != b[i]) mismatch += 1 ;
    return mismatch ;
}

inline std::string stree::Desc(const std::vector<int>& a, unsigned edgeitems ) // static 
{
    std::stringstream ss ;
    ss << "stree::Desc " << a.size() << " : " ;
    for(unsigned i=0 ; i < a.size() ; i++)
    {
        if(i < edgeitems || i > (a.size() - edgeitems) ) ss << a[i] << " " ;
        else if( i == edgeitems ) ss << "... " ;
    }
    std::string s = ss.str();
    return s ;
}

/**
stree::classifySubtrees
------------------------

This is invoked by stree::factorize

Traverse all nodes, computing and collecting subtree digests and adding them to subs_freq
to find the top repeaters.  

**/

inline void stree::classifySubtrees()
{
    if(level>0) std::cout << "[ stree::classifySubtrees " << std::endl ;
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        std::string sub = subtree_digest(nidx) ;
        subs.push_back(sub) ;
        subs_freq->add(sub.c_str());
    }
    if(level>0) std::cout << "] stree::classifySubtrees " << std::endl ;
}


/**
stree::is_contained_repeat
----------------------------

A contained repeat *sub* digest is defined as one where the subtree 
digest of the parent of the first node with *sub* digest passes "pfreq >= FREQ_CUT" 

Note that this definition assumes the first node of a *sub* is representative. 
That might not always be the case. 

Initially tried changing criteria for a contained repeat to be 
that the parent of the first node with the supplied subtree digest 
has a subtree digest frequency equal to that of the original first node, 
but that fails to match GGeo due to a repeats inside repeats recursively 
which do not have the same counts.

Dump ancestry of first sBar::

    SO sBar lvs [8 9]
    lv:8 bb=st.find_lvid_nodes(lv)  bb:[   18    20    22    24    26 ... 65713 65715 65717 65719 65721] b:18 
    b:18 anc=st.get_ancestors(b) anc:[0, 1, 5, 6, 12, 13, 14, 15, 16, 17] 
    st.desc_nodes(anc, brief=True))
    +               snode ix:      0 dh: 0 nc:    2 lv:138. sf 138 :       1 : 8ab45. sWorld0x577e4d0
     +              snode ix:      1 dh: 1 nc:    2 lv: 17. sf 118 :       1 : 6eaf9. sTopRock0x578c0a0
      +             snode ix:      5 dh: 2 nc:    1 lv: 16. sf 120 :       1 : 6db9a. sExpRockBox0x578ce00
       +            snode ix:      6 dh: 3 nc:    3 lv: 15. sf 121 :       1 : 3736e. sExpHall0x578d4f0
        +           snode ix:     12 dh: 4 nc:   63 lv: 14. sf 127 :       1 : f6323. sAirTT0x71a76a0
         +          snode ix:     13 dh: 5 nc:    2 lv: 13. sf  36 :      63 : 66a4f. sWall0x71a8b30
          +         snode ix:     14 dh: 6 nc:    4 lv: 12. sf  35 :     126 : 09a56. sPlane0x71a8bb0
           +        snode ix:     15 dh: 7 nc:    1 lv: 11. sf  32 :     504 : 7d9a6. sPanel0x71a8d90
            +       snode ix:     16 dh: 8 nc:   64 lv: 10. sf  31 :     504 : a1a35. sPanelTape0x71a9090
             +      snode ix:     17 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfb. sBar0x71a9200
    st.desc_nodes([b], brief=True))
              +     snode ix:     18 dh:10 nc:    0 lv:  8. sf   0 :   32256 : 34f45. sBar0x71a9370

CAUTION : that the sf digest counts are for entire geometry, not the subtree of one node. 

The deepest sBar is part of 6 potential repeated instances::

    sWall,sPlane,sPanel,sPanelTape,sBar0x71a9200,sBar0x71a9370.

GGeo picked sPanel (due to repeat candidate cut of 500).

* For now I need to duplicate that here.  

**/

inline bool stree::is_contained_repeat(const char* sub) const
{
    int nidx = get_first(sub);            // first node with this subtree digest  
    int parent = get_parent(nidx);        // parent of first node 
    const char* psub = get_sub(parent) ; 
    int p_freq = subs_freq->get_freq(psub) ; 
    return p_freq >= FREQ_CUT ; 
}

/**
stree::disqualifyContainedRepeats
----------------------------------

Disqualification is not applied during the loop
as that would prevent some disqualifications 
from working as the parents will sometimes
no longer be present. 
 
**/

inline void stree::disqualifyContainedRepeats()
{
    if(level>0) std::cout << "[ stree::disqualifyContainedRepeats " << std::endl ;

    unsigned num = subs_freq->get_num(); 
    std::vector<std::string> disqualify ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = subs_freq->get_key(i);
        int freq = subs_freq->get_freq(i);   // -ve freq means disqualified 
        if(freq < FREQ_CUT) continue ;
        if(is_contained_repeat(sub)) disqualify.push_back(sub) ;       
    }

    subs_freq->set_disqualify( disqualify ); 

    if(level > 0) std::cout 
        << "] stree::disqualifyContainedRepeats " 
        << " disqualify.size " << disqualify.size()
        << std::endl 
        ;
}


/**
stree_subs_freq_ordering
------------------------

Used from stree::sortSubtrees ordering (sub, freq) based on:

1. nidx obtained from stree::get_first(sub:subtree_digest)
2. freq:frequency count 

It is necessary to use two level ordering 
to ensure that same order is achieved 
on different machines. 

**/

struct stree_subs_freq_ordering
{
    const stree* st ; 
    stree_subs_freq_ordering( const stree* st_ ) : st(st_) {} ; 

    bool operator()(
        const std::pair<std::string,int>& a, 
        const std::pair<std::string,int>& b ) const 
    {
        const char* a_sub = a.first.c_str(); 
        const char* b_sub = b.first.c_str(); 
        int a_nidx = st->get_first(a_sub);  
        int b_nidx = st->get_first(b_sub);  
        int a_freq = a.second ; 
        int b_freq = b.second ; 

        return a_freq == b_freq ?  b_nidx > a_nidx : a_freq > b_freq ; 
    }
}; 


/**
stree::sortSubtrees
---------------------

Order the subs_freq (sub,freq) pairs within the vector

**/

inline void stree::sortSubtrees()  // hmm sortSubtreeDigestFreq would be more accurate 
{
    if(level > 0) std::cout << "[ stree::sortSubtrees " << std::endl ;

    stree_subs_freq_ordering ordering(this) ;  
    sfreq::VSU& vsu = subs_freq->vsu ; 
    std::sort( vsu.begin(), vsu.end(), ordering );

    if(level > 0) std::cout << "] stree::sortSubtrees " << std::endl ;
}

/**
stree::enumerateFactors
------------------------

For remaining subs that pass the "freq >= FREQ_CUT"
create sfactor and collect into *factor* vector

**/

inline void stree::enumerateFactors()
{
    if(level > 0) std::cout << "[ stree::enumerateFactors " << std::endl ;
    const sfreq* sf = subs_freq ; 
    unsigned num = sf->get_num(); 
    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = sf->get_key(i); 
        int freq = sf->get_freq(i); 
        if(freq < FREQ_CUT) continue ; 

        sfactor fac ; 
        fac.index = i ; 
        fac.freq = freq ; 
        fac.sensors = 0 ; // set later, from U4Tree::identifySensitiveInstances
        fac.subtree = 0 ; // set later, by stree::labelFactorSubtrees
        fac.set_sub(sub) ;
    
        factor.push_back( fac );  
    }
    if(level > 0) std::cout << "] stree::enumerateFactors " << std::endl ;
}


/**
stree::labelFactorSubtrees
------------------------------

label all nodes of subtrees of all repeats with repeat_index, 
leaving remainder nodes at default of zero repeat_index

**/

inline void stree::labelFactorSubtrees()
{
    int num_factor = factor.size(); 
    if(level>0) std::cout << "[ stree::labelFactorSubtrees num_factor " << num_factor << std::endl ;

    for(int i=0 ; i < num_factor ; i++)
    {
        int repeat_index = i + 1 ;   // leave repeat_index zero for the global remainder 
        sfactor& fac = factor.at(repeat_index-1) ;  // .at is appropriate for small loops 
        std::string sub = fac.get_sub() ;
        assert( fac.index == repeat_index - 1 );  
 
        std::vector<int> outer_node ; 
        get_nodes( outer_node, sub.c_str() ); 
        assert( int(outer_node.size()) ==  fac.freq ); 

        int fac_olvid = -1 ; 
        int fac_subtree = -1 ; 
        for(unsigned i=0 ; i < outer_node.size() ; i++)
        {
            int outer = outer_node[i] ; 

            const snode& ond = nds[outer] ; 

            if( fac_olvid == -1 )
            {
                fac_olvid = ond.lvid ; 
            }
            else
            {
                // all the instances should have the same outer lvid
                assert( ond.lvid == fac_olvid ); 
            }

            std::vector<int> subtree ; 
            get_progeny(subtree, outer); 
            subtree.push_back(outer); 

            if(fac_subtree == -1) 
            {
                fac_subtree = subtree.size() ;  
            }
            else 
            {
                // all the instances must have same number of nodes
                assert( int(subtree.size()) == fac_subtree ); 
            }


            for(unsigned j=0 ; j < subtree.size() ; j++)
            {
                int nidx = subtree[j] ; 
                snode& nd = nds[nidx] ; 
                assert( nd.index == nidx ); 
                nd.repeat_index = repeat_index ; 
                nd.repeat_ordinal = i ;      
            }
        }
        fac.subtree = fac_subtree ; 
        fac.olvid = fac_olvid ; 

        if(level>0) std::cout 
            << "stree::labelFactorSubtrees"
            << fac.desc()
            << " outer_node.size " << outer_node.size()
            << std::endl 
            ; 
    }
    if(level>0) std::cout << "] stree::labelFactorSubtrees " << std::endl ;
}

inline void stree::collectRemainderNodes() 
{
    assert( rem.size() == 0u ); 
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        const snode& nd = nds[nidx] ; 
        assert( nd.index == nidx ); 
        if( nd.repeat_index == 0 ) rem.push_back(nd) ; 
    }
    if(level>0) std::cout << "stree::collectRemainderNodes rem.size " << rem.size() << std::endl ;
}


/**
stree::factorize
-------------------

Canonically invoked from U4Tree::Create

classifySubtrees
   compute and store stree::subtree_digest for subtrees of all nodes

disqualifyContainedRepeats
   flip freq sign in subs_freq to disqualify all contained repeats 

sortSubtrees
   order the subs_freq (sub,freq) pairs within the vector

enumerateFactors
   create sfactor and collect into factor vector 

labelFactorSubtrees
   label all nodes of subtrees of all repeats with repeat_index, 
   leaving remainder nodes at default of zero repeat_index

collectRemainderNodes
   collect global non-instanced 
 
**/

inline void stree::factorize()
{
    if(level>0) std::cout << "[ stree::factorize " << std::endl ;
    classifySubtrees(); 
    disqualifyContainedRepeats();
    sortSubtrees(); 
    enumerateFactors(); 
    labelFactorSubtrees(); 
    collectRemainderNodes(); 

    if(level>0) std::cout << desc_factor() << std::endl ;
    if(level>0) std::cout << "] stree::factorize " << std::endl ;
}


inline int stree::get_num_factor() const
{
    return factor.size(); 
}


inline sfactor& stree::get_factor_(unsigned idx)
{
    assert( idx < factor.size() ); 
    return factor[idx] ; 
}
inline const sfactor& stree::get_factor(unsigned idx) const 
{
    assert( idx < factor.size() ); 
    return factor[idx] ; 
}

inline int stree::get_factor_subtree(unsigned idx) const 
{
    const sfactor& fac = get_factor(idx); 
    return fac.subtree ; 
}
inline int stree::get_factor_olvid(unsigned idx) const 
{
    const sfactor& fac = get_factor(idx); 
    return fac.olvid ; 
}



inline int stree::get_remainder_subtree() const 
{
    return rem.size() ; 
}
inline int stree::get_remainder_olvid() const 
{
    if(rem.size() == 0 ) return -1 ; 
    const snode& out = rem[0] ; 
    return out.lvid ; 
}



inline int stree::get_ridx_subtree(unsigned ridx) const 
{
    return ridx == 0 ? get_remainder_subtree() : get_factor_subtree(ridx - 1 ) ; 
}
inline int stree::get_ridx_olvid(unsigned ridx) const 
{
    return ridx == 0 ? get_remainder_olvid() : get_factor_olvid(ridx - 1 ) ; 
}

inline int stree::get_num_ridx() const 
{
    return 1 + get_num_factor() ; 
}

/**
stree::get_factor_nodes : collect outer volume node indices of *idx* factor (0-based)
--------------------------------------------------------------------------------------

Q: Is factor idx 0 the global remainder OR is it the first instance factor ?
A: First instance (judging by get_num_ridx, get_ridx_olvid) the remainder
   is treated separately from the factors. 

Used by U4Tree::identifySensitiveInstances

1. lookup the subtree digest for factor idx
2. get nodes that match that substree digest

As the factor digests are from the outer volume nodes 
this provides node indices of the outer volumes of 
that the factor. 

**/

inline void stree::get_factor_nodes(std::vector<int>& nodes, unsigned idx) const 
{
    assert( idx < factor.size() ); 
    const sfactor& fac = factor[idx]; 
    std::string sub = fac.get_sub(); 
    int freq = fac.freq ; 

    get_nodes(nodes, sub.c_str() );  

    bool consistent = int(nodes.size()) == freq ; 
    if(!consistent) std::cerr 
        << "stree::get_factor_nodes INCONSISTENCY"
        << " nodes.size " << nodes.size()
        << " freq " << freq 
        << std::endl 
        ;
    assert(consistent );   
}


inline std::string stree::desc_factor() const 
{
    std::stringstream ss ; 
    ss << "stree::desc_factor" << std::endl << sfactor::Desc(factor) ; 
    std::string s = ss.str();
    return s ; 
}




/**
stree::get_repeat_field
-------------------------

::

    In [9]: snode.Label(6,11), f.nds[f.nds[:,ri] == 1 ]
    Out[9]: 
    ('           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      bd',
     array([[194249,      6,  20675,  67846,      2, 194250, 194254,    122, 300000,     -1,     -1,      1,     26],
            [194250,      7,      0, 194249,      2, 194251, 194253,    120,      0,     -1,     -1,      1,     29],
            [194251,      8,      0, 194250,      0,     -1, 194252,    118,      0,     -1,     -1,      1,     36],
            [194252,      8,      1, 194250,      0,     -1,     -1,    119,      0,     -1,     -1,      1,     37],
            [194253,      7,      1, 194249,      0,     -1,     -1,    121,      0,     -1,     -1,      1,     24],

            [194254,      6,  20676,  67846,      2, 194255, 194259,    122, 300001,     -1,     -1,      1,     26],
            [194255,      7,      0, 194254,      2, 194256, 194258,    120,      0,     -1,     -1,      1,     29],
            [194256,      8,      0, 194255,      0,     -1, 194257,    118,      0,     -1,     -1,      1,     36],
            [194257,      8,      1, 194255,      0,     -1,     -1,    119,      0,     -1,     -1,      1,     37],
            [194258,      7,      1, 194254,      0,     -1,     -1,    121,      0,     -1,     -1,      1,     24],

            [194259,      6,  20677,  67846,      2, 194260, 194264,    122, 300002,     -1,     -1,      1,     26],
            [194260,      7,      0, 194259,      2, 194261, 194263,    120,      0,     -1,     -1,      1,     29],
            ...


ADDED ro:repeat_ordinal to snode.h for convenient selecting of the repeats:: 

    In [27]: snode.Label(5,10), f.nds[np.logical_and(f.nds[:,ri] == 2,f.nds[:,ro] == 0)]
    Out[27]: 
    ('          ix     dp     sx     pt     nc     fc     sx     lv     cp     se     sx     ri     ro     bd',
     array([[70979,     6,  3065, 67846,     3, 70980, 70986,   117,     2,    -1,    -1,     2,     0,    26],
            [70980,     7,     0, 70979,     0,    -1, 70981,   111,     0,    -1,    -1,     2,     0,    27],
            [70981,     7,     1, 70979,     0,    -1, 70982,   112,     0,    -1,    -1,     2,     0,    33],
            [70982,     7,     2, 70979,     1, 70983,    -1,   116,     0,    -1,    -1,     2,     0,    29],
            [70983,     8,     0, 70982,     2, 70984,    -1,   115,     0,    -1,    -1,     2,     0,    30],
            [70984,     9,     0, 70983,     0,    -1, 70985,   113,     0,    -1,    -1,     2,     0,    34],
            [70985,     9,     1, 70983,     0,    -1,    -1,   114,     0,    -1,    -1,     2,     0,    35]], dtype=int32))



**/


inline bool stree::SelectNode( const snode& nd, int q_repeat_index, int q_repeat_ordinal ) // static
{
    bool all_ordinal = q_repeat_ordinal == -2 ; 
    bool select = all_ordinal ? 
                                nd.repeat_index == q_repeat_index 
                              :
                                nd.repeat_index == q_repeat_index && nd.repeat_ordinal == q_repeat_ordinal
                              ;

    return select ;  
}

inline void stree::get_repeat_field(std::vector<int>& result, char q_field , int q_repeat_index, int q_repeat_ordinal ) const 
{
    int num_nd = nds.size() ; 

    int nidx_mismatch = 0 ; 

    for(int nidx=0 ; nidx < num_nd ; nidx++)
    {
        const snode& nd = nds[nidx] ; 

        if(nd.index != nidx) 
        {
            std::cerr 
               << "stree::get_repeat_field"
               << " ERROR : NIDX MISMATCH : "
               << " nidx " << nidx
               << " nd.index " << nd.index
               << " num_nd " << num_nd
               << std::endl 
               ;
            nidx_mismatch += 1 ; 
        }

        if(SelectNode(nd, q_repeat_index, q_repeat_ordinal))
        {
            int field = -3 ; 
            switch(q_field)
            {
                case 'I': field = nd.index ; break ; 
                case 'L': field = nd.lvid  ; break ; 
            }
            result.push_back(field) ; 
        }
    }

    if( nidx_mismatch > 0 )
    {
        std::cerr
            << "stree::get_repeat_field"
            << " FATAL : NIDX MISMATCH : "
            << " nidx_mismatch " << nidx_mismatch
            << " num_nd " << num_nd
            << " q_field " << q_field 
            << " q_repeat_index " << q_repeat_index
            << " q_repeat_ordinal " << q_repeat_ordinal
            << std::endl 
            ; 
    }
    assert( nidx_mismatch == 0 ); 
}


inline void stree::get_repeat_lvid(std::vector<int>& lvids, int q_repeat_index, int q_repeat_ordinal ) const 
{
    get_repeat_field(lvids, 'L', q_repeat_index, q_repeat_ordinal );  
}
inline void stree::get_repeat_nidx(std::vector<int>& nidxs, int q_repeat_index, int q_repeat_ordinal ) const 
{
    get_repeat_field(nidxs, 'I', q_repeat_index, q_repeat_ordinal );  
}
inline void stree::get_remainder_nidx(std::vector<int>& nodes ) const 
{
    int q_repeat_index = 0 ; 
    int q_repeat_ordinal = -2 ; 
    get_repeat_nidx(nodes, q_repeat_index, q_repeat_ordinal); 
}


inline void stree::get_repeat_node(std::vector<snode>& nodes, int q_repeat_index, int q_repeat_ordinal ) const 
{
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        const snode& nd = nds[nidx] ; 
        assert( nd.index == nidx ); 
        if(SelectNode(nd, q_repeat_index, q_repeat_ordinal)) nodes.push_back(nd); 
    }
}



inline std::string stree::desc_repeat_nodes() const 
{
    int num_factor = factor.size(); 
    std::stringstream ss ; 
    ss << "stree::desc_repeat_nodes"
       << " num_factor " << num_factor 
       << std::endl 
       ;  

    int total = 0 ; 
    int remainder = 0 ; 

    for(int i=0 ; i < num_factor + 1 ; i++)   // NB num_factor+1 : remainder in i=0 
    {
        int q_ridx = i ;  
        std::vector<int> nidxs ; 
        get_repeat_nidx(nidxs, q_ridx ); 

        int num_nodes = nidxs.size() ; 
        if(q_ridx == 0) remainder = num_nodes ; 

        total += num_nodes ;  
        ss 
            << " q_ridx " << std::setw(3) << q_ridx 
            << " num_nodes " << std::setw(8) << num_nodes 
            << std::endl 
            ; 
    }

    ss 
        << std::setw(30) << " total     " 
        << std::setw(8)  << total  
        << std::endl 
        << std::setw(30) << " remainder " 
        << std::setw(8)  << remainder  
        << std::endl  
        << std::setw(30) << " total - remainder " 
        << std::setw(8)  << (total - remainder) 
        << std::endl 
        ; 

    std::string s = ss.str(); 
    return s ; 
}



/**
stree::add_inst
----------------

Canonically invoked from U4Tree::Create 

1. lookup snode from nidx
2. encode snode identity info into transform fourth columns
3. collect transforms into inst, iinst vectors and nidx into inst_nidx

* NB restrict to 32 bit of identity info, so it survives narrowing 

**/

inline void stree::add_inst( 
    glm::tmat4x4<double>& tr_m2w,  
    glm::tmat4x4<double>& tr_w2m, 
    int gas_idx, 
    int nidx )
{
    assert( nidx > -1 && nidx < int(nds.size()) ); 
    const snode& nd = nds[nidx];    // structural volume node

    int ins_idx = int(inst.size()); // follow sqat4.h::setIdentity

    glm::tvec4<int64_t> col3 ;   // formerly uint64_t 

    col3.x = ins_idx ;            // formerly  +1 
    col3.y = gas_idx ;            // formerly  +1 
    col3.z = nd.sensor_id ;       // formerly ias_idx + 1 (which was always 1)
    col3.w = nd.sensor_index ; 

    strid::Encode(tr_m2w, col3 );
    strid::Encode(tr_w2m, col3 );
 
    inst.push_back(tr_m2w);
    iinst.push_back(tr_w2m);

    inst_nidx.push_back(nidx); 
}

/**
stree::add_inst
------------------


::

    In [7]: f.inst_f4[:,:,3].view(np.int32)
    Out[7]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  18116],
           [     2,      1, 300001,  18117],
           [     3,      1, 300002,  18118],
           [     4,      1, 300003,  18119],
           ...
           [ 48472,      9,      3,    499],
           [ 48473,      9,      0,    500],
           [ 48474,      9,      1,    501],
           [ 48475,      9,      2,    502],
           [ 48476,      9,      3,    503]], dtype=int32)

          # ins_idx, gas_idx,  sid    six  

    In [8]: f.inst_f4.shape
    Out[8]: (48477, 4, 4)

    In [10]: f.inst_nidx
    Out[10]: 
    array([     0, 279688, 279693, 279698, 279703, 279708, 279713, 279718, 279723, 279728, 279733, 279738, 279743, 279748, 279753, 279758, ...,  63638,  63768,  63898,  64028,  64159,  64289,  64419,
            64549,  64681,  64811,  64941,  65071,  65202,  65332,  65462,  65592], dtype=int32)

    In [11]: f.inst_nidx.shape
    Out[11]: (48477,)



**/


inline void stree::add_inst() 
{
    glm::tmat4x4<double> tr_m2w(1.) ; 
    glm::tmat4x4<double> tr_w2m(1.) ; 
    add_inst(tr_m2w, tr_w2m, 0, 0 );   // global instance with identity transforms 

    unsigned num_factor = get_num_factor(); 
    for(unsigned i=0 ; i < num_factor ; i++)
    {
        std::vector<int> nodes ; 
        get_factor_nodes(nodes, i);  

        unsigned gas_idx = i + 1 ; // 0 is the global instance, so need this + 1  
        std::cout 
            << "stree::add_inst"
            << " i " << std::setw(3) << i 
            << " gas_idx " << std::setw(3) << gas_idx
            << " nodes.size " << std::setw(7) << nodes.size()
            << std::endl 
            ;

        for(unsigned j=0 ; j < nodes.size() ; j++)
        {
            int nidx = nodes[j]; 
            //get_m2w_product(tr_m2w, nidx, false); 
            //get_w2m_product(tr_w2m, nidx, true ); 

            bool local = false ; 
            bool reverse = false ; 
            get_node_product( tr_m2w, tr_w2m, nidx, local, reverse, nullptr  ); 

            add_inst(tr_m2w, tr_w2m, gas_idx, nidx ); 
        }
    }
    narrow_inst(); 
}

inline void stree::narrow_inst()
{
    strid::Narrow( inst_f4,   inst ); 
    strid::Narrow( iinst_f4, iinst ); 
}

inline void stree::clear_inst() 
{
    inst.clear(); 
    iinst.clear(); 
    inst_f4.clear(); 
    iinst_f4.clear(); 
}

inline std::string stree::desc_inst() const 
{
    std::stringstream ss ; 
    ss << "stree::desc_inst"
       << " inst " << inst.size()
       << " iinst " << iinst.size()
       << " inst_f4 " << inst_f4.size()
       << " iinst_f4 " << iinst_f4.size()
       ;
    std::string s = ss.str(); 
    return s ; 
}

/**
stree::get_mtindex_range
--------------------------

As the 0-based G4Material index is a creation index and not all 
created G4Material may be be in active use there can
be gaps in the range of this origin material index. 

**/

inline void stree::get_mtindex_range(int& mn, int& mx ) const 
{
    mn = std::numeric_limits<int>::max() ; 
    mx = 0 ; 
    for(unsigned i=0 ; i < mtindex.size() ; i++) 
    {
        int mtidx = mtindex[i] ;  
        if(mtidx > mx) mx = mtidx ; 
        if(mtidx < mn) mn = mtidx ; 
    }
}

inline std::string stree::desc_mt() const 
{
    int mn, mx ; 
    get_mtindex_range( mn, mx);  
    std::stringstream ss ; 
    ss << "stree::desc_mt"
       << " mtname " << mtname.size()
       << " mtname_no_rindex " << mtname_no_rindex.size()
       << " mtindex " << mtindex.size()
       << " mtline " << mtline.size()
       << " mtindex.mn " << mn 
       << " mtindex.mx " << mx 
       << std::endl 
       ;

    // populate mtline with SBnd::FillMaterialLine after have bndnames

    bool with_mtline = mtline.size() > 0 ; 
    assert( mtname.size() == mtindex.size() ); 
    unsigned num_mt = mtname.size(); 

    if(with_mtline) assert( mtline.size() == num_mt ); 

    for(unsigned i=0 ; i < num_mt ; i++)
    {
        const char* mtn = mtname[i].c_str(); 
        int         mtidx = mtindex[i];
        int         mtlin = with_mtline ? mtline[i] : -1 ; 
        ss 
            << " i " << std::setw(3) << i  
            << " mtindex " << std::setw(3) << mtidx  
            << " mtline " << std::setw(3)  << mtlin  
            << " mtname " << mtn 
            << std::endl 
            ;
    }

    std::string str = ss.str(); 
    return str ; 
}

/**
stree::desc_bd
----------------

**/

inline std::string stree::desc_bd() const 
{
    std::stringstream ss ; 
    ss << "stree::desc_bd"
       << " vbd.size " << vbd.size()
       << " bdname.size " << bdname.size()
       << std::endl 
       ; 

    assert( vbd.size() == bdname.size() ); 

    int num_bd = vbd.size() ; 
    for(int i=0 ; i < num_bd ; i++)
    {
        const std::string& bdn = bdname[i] ; 
        const int4& bd_ = vbd[i] ;  
        ss << std::setw(4) << i 
           << "("
           << std::setw(3) << bd_.x 
           << " "
           << std::setw(3) << bd_.y 
           << " "
           << std::setw(3) << bd_.z 
           << " "
           << std::setw(3) << bd_.w 
           << ") "
           << bdn
           << std::endl 
           ; 
    }
    std::string str = ss.str(); 
    return str ; 
}





/**
stree::add_material
----------------------

Canonically called from U4Tree::initMaterials_r/U4Tree::initMaterial

g4index is the Geant4 creation index obtained from G4Material::GetIndex

Note that not all G4Material are added, only G4Material that are 
referenced from G4LogicalVolume are added, 
so the g4index might not match the idx from mtname. 

**/

inline int stree::add_material( const char* name, unsigned g4index )
{
    int idx = mtname.size() ; 
    mtname.push_back(name); 
    mtindex.push_back(g4index); 
    // assert( idx == g4index );   NOT FULFILLED
    return idx ; 
} 
inline int stree::num_material() const 
{
    return mtname.size();  
}

inline int stree::add_surface( const char* name )
{
    int idx = -1 ; 
    int prior = stree::GetValueIndex<std::string>( suname, name ) ; 
    if(prior > -1)  
    {   
        idx = prior ; 
    }
    else   // new surface name
    {
        idx = suname.size() ; 
        suname.push_back(name) ;   
        suindex.push_back(idx); 
        int idx2 = stree::GetValueIndex<std::string>( suname, name ) ; 
        assert( idx2 == idx ); 
    }   
    return idx ; 
} 
inline int stree::get_surface( const char* name ) const 
{
    return stree::GetValueIndex<std::string>(suname, name ) ; 
}
inline int stree::num_surface() const 
{
    return suname.size();  
}


inline int stree::add_surface(const std::vector<std::string>& names  )
{
    int idx = -1 ; 
    for(unsigned i=0 ; i < names.size() ; i++) 
    {    
        const char* sn = names[i].c_str() ; 
        idx = add_surface( sn ); 
    }    
    return idx ; 
} 



/**
stree::add_surface_implicit
----------------------------

Used from U4TreeBorder::get_implicit_idx

THIS FORMERLY RETURNED THE IMPLICIT_IDX, 
NOW RETURNING THE STANDARD SURFACE IDX 

**/


inline int stree::add_surface_implicit( const char* name )
{
    int idx = add_surface(name);   

    int implicit_idx = stree::GetValueIndex<std::string>( implicit, name ) ; 
    if(implicit_idx == -1)  // new implicit 
    {   
        implicit.push_back(name) ;   
        implicit_idx = stree::GetValueIndex<std::string>( implicit, name ) ; 
        assert( implicit_idx > -1 );  
        assert( idx == num_surface_standard() + implicit_idx ) ; 
    }   
    
    return idx ; 
}


inline int stree::get_surface_implicit( const char* name ) const 
{
    return stree::GetValueIndex<std::string>(implicit, name ) ; 
}
inline int stree::num_surface_implicit() const 
{
    return implicit.size();  
}
inline int stree::num_surface_standard() const 
{
    return num_surface() - num_surface_implicit() ;  
}







inline int stree::add_boundary( const int4& bd_ )
{
    int boundary = GetValueIndex<int4>( vbd, bd_ ) ; 
    if(boundary == -1)  // new boundary 
    {
        int num_bd_check = vbd.size();  
        std::string bdn = get_boundary_name(bd_,'/') ; 
        vbd.push_back(bd_) ; 
        bdname.push_back(bdn.c_str()) ; 
        boundary = GetValueIndex<int4>( vbd, bd_ ) ; 
        assert( num_bd_check == boundary );  
    }
    return boundary ; 
}


inline const char* stree::get_material_name( int idx ) const 
{
    return snam::get(mtname, idx) ; 
}
inline const char* stree::get_surface_name( int idx ) const 
{
    return snam::get(suname, idx) ; 
}
inline std::string stree::get_boundary_name( const int4& bd_, char delim ) const 
{
    const char* omat = get_material_name( bd_.x ); 
    const char* osur = get_surface_name(  bd_.y ); 
    const char* isur = get_surface_name(  bd_.z ); 
    const char* imat = get_material_name( bd_.w ); 

    assert( omat ); 
    assert( imat ); 

    std::stringstream ss ;
    ss   
       << omat 
       << delim
       << ( osur ? osur : "" ) 
       << delim 
       << ( isur ? isur : "" ) 
       << delim
       << imat 
       ;
    
    std::string str = ss.str(); 
    return str ; 
}



/**
stree::get_surface_subfold
---------------------------

Note that implicit and perfect names will be found
in the suname vector but there is no corresponding 
subfold for them, hence no metadata. 

**/
inline NPFold* stree::get_surface_subfold(int idx) const 
{
    const char* sn = get_surface_name(idx); 
    assert(sn) ; 
    NPFold* sub = surface->get_subfold(sn) ;  
    return sub ; 
}

/**
stree::initStandard
----------------------

Called from U4Tree::initStandard after most of 
the conversion is done with mat and sur arrays prepared. 

**/

inline void stree::initStandard()
{
    standard->deferred_init( vbd, bdname, suname, surface ); 
}


/**
stree::import_bnd
-------------------

Moved from SSim::import_bnd 

**/

inline void stree::import_bnd(const NP* bnd)
{
    assert(bnd) ; 
    const std::vector<std::string>& bnames = bnd->names ; 

    assert( mtline.size() == 0 ); 
    assert( mtname.size() == mtindex.size() ); 

    // for each mtname use bnd->names to fill the mtline vector
    SBnd::FillMaterialLine( mtline, mtindex, mtname, bnames ); 

    // fill (int,int) map from the mtline and mtindex vectors 
    init_mtindex_to_mtline() ; 
    
    if( level > 1 ) std::cerr 
        << "stree::import_bnd" 
        << " level > 1 [" << level << "]" 
        << " bnd " << bnd->sstr()
        << " desc_mt " 
        << std::endl 
        << desc_mt() 
        << std::endl 
        ; 
}


/**
stree::init_mtindex_to_mtline
------------------------------

Canonically invoked from SSim::import_bnd/SBnd::FillMaterialLine following 
live creation or from stree::load_ when loading a persisted stree.  

TODO: suspect this complication can be avoided using NPX.h load/save of maps ?
**/

inline void stree::init_mtindex_to_mtline()
{
    bool consistent = mtindex.size() == mtline.size() ;
    if(!consistent) std::cerr 
        << "stree::init_mtindex_to_mtline"
        << " mtindex.size " << mtindex.size()
        << " mtline.size " << mtline.size()
        << " : must use SBnd::FillMaterialLine once have bnd specs" 
        << std::endl 
        ;
 
    assert(consistent); 
    for(unsigned i=0 ; i < mtindex.size() ; i++) mtindex_to_mtline[mtindex[i]] = mtline[i] ;  
}

inline int stree::lookup_mtline( int mtindex ) const 
{
    return mtindex_to_mtline.count(mtindex) == 0 ? -1 :  mtindex_to_mtline.at(mtindex) ;  
}


