#pragma once
/**
stree.h : minimal representation of the structural geometry tree
=====================================================================

This is exploring a minimal approach to geometry translation

* u4/U4Tree.h populates stree.h from traversals of Geant4 volumes.
* stree.h replaces lots of GGeo code, notably: GInstancer.cc GNode.cc

* mat/sur/bnd materials/surfaces/boundaries are handled by sstandard

Example of stree.h population
------------------------------

~/o/u4/tests/U4TreeCreateTest.sh


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

    NB : THIS IS UNHOLY MIX OF OLD AND NEW : NOW  REPLACED

ggeo/tests/GGeoLoadFromDirTest.cc
    dev of the interim stree GGeo integration for sensor info

    NOW REPLACED

ggeo/GGeo.cc
    GGeo:m_tree with setTree/getTree : but treated as foreign member, only GGeo::save saves it
    this m_tree was used for debugging a now resolved discrepancy between X4+GGeo and U4Test
    transforms : that is suspected but not confirmed to have been caused by a
    stree parent pointer bug

    NOW REPLACED

extg4/X4PhysicalVolume.cc
    X4PhysicalVolume::convertStructure creates stree.h and setTree into GGeo
    X4PhysicalVolume::convertStructure_r collects snode.h and transforms into the GGeo/stree

    NOW REPLACED

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


mapping stree.h/nidx to CSGFoundry/globalPrimIdx ?
----------------------------------------------------

see populate_prim_nidx


mapping for the remainder non-instanced volumes
-------------------------------------------------

For the global remainder "instance" things are not so straightforward as there is only
one of them with an identity transform and the nodes within it are not contiguous, they
are what is left when all the repeated subtrees have been removed : so it will
start from root nidx:0 and will have lots of gaps.

Actually the natural place to keep "map-back" info for the remainder
is withing the CSGPrim.  Normally use of that is for identity purposes is restricted
because tthe CSGPrim are references from all the instances but for the remainder
the CSGPrim are only referenced once.

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
#include <csignal>
#include <vector>
#include <string>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "NP.hh"
#include "NPX.h"
#include "NPFold.h"
#include "OpticksCSG.h"

#include "ssys.h"
#include "sstr.h"
#include "scuda.h"
#include "snode.h"
#include "sdigest.h"
#include "sfreq.h"
#include "sstr.h"
#include "strid.h"
#include "sfactor.h"
#include "stran.h"
#include "stra.h"
#include "slist.h"

#include "s_csg.h"
#include "sn.h"

#include "stra.h"
#include "sstandard.h"
#include "smatsur.h"
#include "snam.h"
#include "SBnd.h"
#include "SCenterExtentFrame.h"

// transitional ?
#include "sframe.h"



struct stree_standin
{
    std::vector<snode> nds ;               // snode info for all structural nodes, the volumes
    std::vector<glm::tmat4x4<double>> m2w ; // model2world transforms for all nodes
    std::vector<glm::tmat4x4<double>> gtd ; // GGeo Transform Debug, added from X4PhysicalVolume::convertStructure_r
};


struct stree
{
    static constexpr const char* stree__populate_prim_nidx = "stree__populate_prim_nidx" ;
    static constexpr const char* stree__populate_nidx_prim = "stree__populate_nidx_prim" ;

    static constexpr const char* _EXTENT_PFX = "EXTENT:" ;
    static constexpr const char* stree__force_triangulate_solid = "stree__force_triangulate_solid" ;
    static constexpr const char* stree__get_frame_dump = "stree__get_frame_dump" ;

    static constexpr const int MAXDEPTH = 15 ; // presentational limit only
    static constexpr const int FREQ_CUT = 500 ;   // HMM GInstancer using 400
    // subtree digests with less repeats than FREQ_CUT within the entire geometry
    // are not regarded as repeats for instancing factorization purposes

    static constexpr const char* BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ;  // default
    static constexpr const char* RELDIR = "stree" ;

    static constexpr const char* NDS = "nds.npy" ;
    static constexpr const char* NDS_NOTE = "snode.h structural volume nodes" ;
    static constexpr const char* REM = "rem.npy" ;
    static constexpr const char* TRI = "tri.npy" ;
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

    static constexpr const char* FORCE_TRIANGULATE_LVID = "force_triangulate_lvid.npy" ;


#ifdef DEBUG_IMPLICIT
    static constexpr const char* IMPLICIT_ISUR = "implicit_isur.npy" ;
    static constexpr const char* IMPLICIT_OSUR = "implicit_osur.npy" ;
#endif


    //static constexpr const char* SUINDEX = "suindex.npy" ;

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
    static constexpr const char* MESH = "mesh" ;
    static constexpr const char* STANDARD = "standard" ;


    static constexpr const char* INST = "inst.npy" ;
    static constexpr const char* IINST = "iinst.npy" ;
    static constexpr const char* INST_F4 = "inst_f4.npy" ;
    static constexpr const char* IINST_F4 = "iinst_f4.npy" ;

    static constexpr const char* SENSOR_ID = "sensor_id.npy" ;
    static constexpr const char* SENSOR_NAME = "sensor_name.npy" ;
    static constexpr const char* MTINDEX_TO_MTLINE = "mtindex_to_mtline.npy" ;

    static constexpr const char* INST_INFO = "inst_info.npy" ;
    static constexpr const char* INST_NIDX = "inst_nidx.npy" ;

    static constexpr const char* PRIM_NIDX = "prim_nidx.npy" ;
    static constexpr const char* NIDX_PRIM = "nidx_prim.npy" ;



    int level ;                            // verbosity
    const char*      force_triangulate_solid ;
    std::vector<int> force_triangulate_lvid ;
    bool get_frame_dump ;


    std::vector<std::string> mtname ;       // unique material names
    std::vector<std::string> mtname_no_rindex ;
    std::vector<int>         mtindex ;      // G4Material::GetIndex 0-based creation indices
    std::vector<int>         mtline ;
    std::map<int,int>        mtindex_to_mtline ;   // filled from mtindex and mtline via init_material_mapping

    std::vector<std::string> suname_raw ;   // surface names, direct from Geant4
    std::vector<std::string> suname ;       // surface names
    //std::vector<int>         suindex ;      // HMM: is this needed, its just 0,1,2,...
    std::vector<int4>        vbd ;
    std::vector<std::string> bdname ;
    std::vector<std::string> implicit ;  // names of implicit surfaces

    std::vector<std::string> soname_raw ;   // solid names, my have 0x pointer suffix
    std::vector<std::string> soname ;       // unique solid names, created with sstr::StripTail_Unique with _1 _2 ... uniqing
    std::vector<sn*>         solids ;       // used from U4Tree::initSolid but not available postcache, instead use sn::Get methods

    std::vector<glm::tmat4x4<double>> m2w ; // local (relative to parent) "model2world" transforms for all nodes
    std::vector<glm::tmat4x4<double>> w2m ; // local (relative to parent( "world2model" transforms for all nodes
    std::vector<glm::tmat4x4<double>> gtd ; // global (relative to root) "GGeo Transform Debug" transforms for all nodes
    // "gtd" formerly from X4PhysicalVolume::convertStructure_r

    std::vector<snode> nds ;               // snode info for all structural nodes, the volumes
    std::vector<snode> rem ;               // subset of nds with the remainder nodes
    std::vector<snode> tri ;               // subset of nds which are configured to be force triangulated (expected to otherwise be remainder nodes)
    std::vector<std::string> digs ;        // per-node digest for all nodes
    std::vector<std::string> subs ;        // subtree digest for all nodes
    std::vector<sfactor> factor ;          // small number of unique subtree factor, digest and freq

    std::vector<int> sensor_id ;           // updated by reorderSensors
    unsigned sensor_count ;
    std::vector<std::string> sensor_name ;


    sfreq* subs_freq ;                     // occurence frequency of subtree digests in entire tree
                                           // subs are collected in stree::classifySubtrees

    s_csg* _csg ;                          // sn.h based csg node trees

    sstandard* standard ;                  // mat/sur/bnd/bd/optical/wavelength/energy/rayleigh

    NPFold* material ;   // material properties from G4 MPTs
    NPFold* surface ;    // surface properties from G4 MPTs, includes OpticalSurfaceName osn in metadata
    NPFold* mesh ; // triangulation of all solids
    const char* MOI ;


    std::vector<glm::tmat4x4<double>> inst ;
    std::vector<glm::tmat4x4<float>>  inst_f4 ;
    std::vector<glm::tmat4x4<double>> iinst ;
    std::vector<glm::tmat4x4<float>>  iinst_f4 ;

    std::vector<int4>                 inst_info ;
    std::vector<int>                  inst_nidx ;

    std::vector<int>                  prim_nidx ; // experimental: see populate_prim_nidx
    std::vector<int>                  nidx_prim ; // experimental: see populate_nidx_prim

    stree();

    void init();
    void set_level(int level_);

    std::string desc() const ;
    std::string desc_soname() const ;
    std::string desc_lvid() const ;
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

    const std::vector<snode>* get_node_vector( char _src ) const ; // 'N':nds 'R':rem 'T':tri
    void find_lvid_nodes_( std::vector<snode>& nodes, int lvid, char _src ) const ;
    void find_lvid_nodes(  std::vector<int>& nodes, int lvid, char _src ) const ;
    int count_lvid_nodes( int lvid, char _src='N' ) const ;

    void find_lvid_nodes( std::vector<int>& nodes, const char* soname_, bool starting ) const ;
    int  find_lvid_node( const char* q_soname, int ordinal ) const ;
    int  find_lvid_node( const char* q_spec ) const ; // eg HamamatsuR12860sMask_virtual:0:1000



    const snode* pick_lvid_ordinal_node( int lvid, int ordinal, char ridx_type ) const ;
    const snode* _pick_lvid_ordinal_node( int lvid, int ordinal, char ridx_type ) const ;

    int pick_lvid_ordinal_repeat_ordinal_inst_( int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
    int parse_spec(int& lvid, int& lvid_ordinal, int& repeat_ordinal, const char* q_spec ) const ;
    int pick_lvid_ordinal_repeat_ordinal_inst( const char* q_spec ) const ;


   // transitional method for matching with CSGFoundry::getFrame
    void get_frame_f4( sframe& fr, int idx ) const ;


    sfr  get_frame_moi() const ;
    sfr  get_frame(const char* q_spec) const ;
    bool has_frame(const char* q_spec) const ;


    int get_frame_instanced(  sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;

    int get_frame_remainder(  sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
    int get_frame_triangulate(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
    int get_frame_global(     sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
    int _get_frame_global(     sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal, char ridx_type ) const ;

    int get_node_ce_bb(  std::array<double,4>& ce , std::array<double,6>& bb, const snode& node ) const ;
    int get_node_bb(     std::array<double,6>& bb , const snode& node ) const ;


    void get_sub_sonames( std::vector<std::string>& sonames ) const ;
    const char* get_sub_soname(const char* sub) const ;

    static std::string Name( const std::string& name, bool strip );
    std::string get_lvid_soname(int lvid, bool strip ) const ;
    const std::string& get_lvid_soname_(int lvid) const ;

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




    void get_combined_transform(
             glm::tmat4x4<double>& t,
             glm::tmat4x4<double>& v,
             const snode& node,
             const sn* nd,
             std::ostream* out
             ) const ;

    std::string desc_combined_transform(
             glm::tmat4x4<double>& t,
             glm::tmat4x4<double>& v,
             const snode& node,
             const sn* nd
             ) const ;

    const Tran<double>* get_combined_tran_and_aabb(
             double* aabb,
             const snode& node,
             const sn* nd,
             std::ostream* out
             ) const ;

    void get_transformed_aabb(
             double* aabb,
             const snode& node,
             const sn* nd,
             std::ostream* out
             ) const ;

    void get_prim_aabb( double* aabb, const snode& node, std::ostream* out ) const ;

    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ;


    std::string subtree_digest( int nidx ) const ;
    static std::string depth_spacer(int depth);

    std::string desc_node_(int nidx, const sfreq* sf ) const ;
    std::string desc_node(int nidx, bool show_sub=false) const ;

    std::string desc_nodes( const std::vector<int>&   nn, int edgeitems=10) const ;
    std::string desc_nodes_(const std::vector<snode>& nn, int edgeitems=10) const ;
    std::string desc_node_solids() const ;
    std::string desc_solids() const ;
    std::string desc_solid(int lvid) const ;


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

    static void FindForceTriangulateLVID(std::vector<int>& lvid, const std::vector<std::string>& _sonames, const char* _force_triangulate_solid, char delim=','  );
    std::string descForceTriangulateLVID() const ;
    bool        is_force_triangulate( int lvid ) const ;


    void classifySubtrees();
    bool is_contained_repeat(const char* sub) const ;
    void disqualifyContainedRepeats();
    void sortSubtrees();
    void enumerateFactors();
    void labelFactorSubtrees();
    void findForceTriangulateLVID();
    void collectGlobalNodes();
    std::string descNodes() const;

    static constexpr const char* _findForceTriangulateLVID_DUMP = "stree__findForceTriangulateLVID_DUMP" ;

    void factorize();

    int get_num_factor() const ;
    sfactor& get_factor_(unsigned idx) ;
    const sfactor& get_factor(unsigned idx) const ;

    int      get_factor_subtree(unsigned idx) const ;
    int      get_factor_olvid(unsigned idx) const ;   // outer-lv-id

    int      get_remainder_subtree() const ;
    int      get_triangulate_subtree() const ;
    int      get_remainder_olvid() const  ;
    int      get_triangulate_olvid() const  ;

    int      get_ridx_subtree(unsigned ridx) const ;
    int      get_ridx_olvid(  unsigned ridx) const ;

    int      get_num_ridx() const ;
    int      get_num_ridx_(char ridx_type) const ;
    int      get_num_remainder() const ;
    int      get_num_triangulated() const ;
    char     get_ridx_type(int ridx) const ;

    void get_factor_nodes(std::vector<int>& nodes, unsigned idx) const ;
    std::string desc_factor_nodes(unsigned idx) const ;
    std::string desc_factor() const ;

    static bool SelectNode( const snode& nd, int q_repeat_index, int q_repeat_ordinal );
    // q_repeat_ordinal:-2 selects all repeat_ordinal

    void get_repeat_field(std::vector<int>& result, char q_field , int q_repeat_index, int q_repeat_ordinal ) const ;
    void get_repeat_lvid( std::vector<int>& lvids, int q_repeat_index, int q_repeat_ordinal=-2 ) const ;
    void get_repeat_nidx( std::vector<int>& nidxs, int q_repeat_index, int q_repeat_ordinal=-2 ) const ;

    void get_remainder_nidx(std::vector<int>& nidxs ) const ;

    void get_repeat_node( std::vector<snode>& nodes, int q_repeat_index, int q_repeat_ordinal ) const ;
    std::string desc_repeat_node(int q_repeat_index, int q_repeat_ordinal) const ;

    std::string desc_repeat_nodes() const ;


    void add_inst( glm::tmat4x4<double>& m2w, glm::tmat4x4<double>& w2m, int gas_idx, int nidx );
    void add_inst_identity( int gas_idx, int nidx );
    void add_inst();

    void narrow_inst();
    void clear_inst();
    std::string desc_inst() const ;
    std::string desc_inst_info() const ;
    std::string desc_inst_info_check() const;

    int find_inst_gas(        int q_gas_idx, int q_gas_ordinal ) const ;
    int find_inst_gas_slowly( int q_gas_idx, int q_gas_ordinal ) const ;
    void find_inst_gas_slowly_( std::vector<int>& v_inst_idx , int q_gas_idx ) const ;


    const glm::tmat4x4<double>* get_inst(int idx) const ;
    const glm::tmat4x4<double>* get_iinst(int idx) const ;
    const glm::tmat4x4<float>*  get_inst_f4(int idx) const ;
    const glm::tmat4x4<float>*  get_iinst_f4(int idx) const ;


    void get_mtindex_range(int& mn, int& mx ) const ;
    std::string desc_mt() const ;
    std::string desc_bd() const ;

    void initStandard() ;

    static constexpr const char* _init_material_mapping_DUMP = "stree__init_material_mapping_DUMP" ;
    void init_material_mapping();

    int add_material( const char* name, unsigned g4index );
    int num_material() const ;

    int add_extra_surface( const char* name );
    int add_extra_surface( const std::vector<std::string>& names  );

    int get_surface( const char* name ) const ;
    int num_surface() const ;   // total including implicit


    int add_surface_implicit( const char* name );
    int get_surface_implicit( const char* name ) const ;
    int num_surface_implicit() const ;
    int num_surface_standard() const ; // total with implicit subtracted


    int add_boundary( const int4& bd_ );

    const char* get_material_name(int idx) const ;
    const char* get_surface_name(int idx) const ;
    std::string get_boundary_name( const int4& bd_, char delim ) const ;

    NPFold* get_surface_subfold(int idx) const ;

    //void import_bnd(const NP* bnd);
    void init_mtindex_to_mtline();
    int lookup_mtline( int mtindex ) const ;

    // experimental
    void populate_prim_nidx();
    void faux_importSolid();
    void faux_importSolidGlobal(int ridx, char ridx_type);
    void faux_importSolidFactor(int ridx, char ridx_type);
    int  faux_importPrim(int primIdx, const snode& node );
    int  search_prim_for_nidx_first(int nidx) const ;

    void populate_nidx_prim();
    int  get_prim_for_nidx(int nidx) const ;

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

HMM the force_triangulate_solid envvar only relevant for stree creation, not with loaded stree ?

**/


inline stree::stree()
    :
    level(ssys::getenvint("stree_level", 0)),
    force_triangulate_solid(ssys::getenvvar(stree__force_triangulate_solid,nullptr)),
    get_frame_dump(ssys::getenvbool(stree__get_frame_dump)),
    sensor_count(0),
    subs_freq(new sfreq),
    _csg(new s_csg),
    standard(new sstandard),
    material(new NPFold),
    surface(new NPFold),
    mesh(new NPFold),
    MOI(ssys::getenvvar("MOI", "0:0:-1"))
{
    init();
}

inline void stree::init()
{
    if(level > 0) std::cout
         << "stree::init "
         << " force_triangulate_solid [" << ( force_triangulate_solid ? force_triangulate_solid : "-" ) << "]"
         << std::endl
         ;
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
       << " suname_raw " << suname_raw.size() << div
       << " suname " << suname.size() << div
       //<< " suindex " << suindex.size() << div
       << " vbd " << vbd.size() << div
       << " bdname " << bdname.size() << div
       << " implicit " << implicit.size() << div
       << " soname " << soname.size() << div
       << " force_triangulate_lvid " << force_triangulate_lvid.size() << div
       << " solids " << solids.size() << div
       << " sensor_count " << sensor_count << div
       << " nds " << nds.size() << div
       << " rem " << rem.size() << div
       << " tri " << tri.size() << div
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
       << desc_lvid()
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
       << " stree::desc.mesh "
       << std::endl
       << ( mesh ? mesh->desc() : "-" )
       << std::endl
       << " stree::desc.csg "
       << std::endl
       << ( _csg ? _csg->desc() : "-" )
       << std::endl
       << "]stree::desc"
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}


inline std::string stree::desc_soname() const
{
    std::stringstream ss ;
    ss << "[stree::desc_soname\n" ;
    for(int i=0 ; i < int(soname.size()) ; i++) ss << "[" << soname[i] << "]\n" ;
    ss << "]stree::desc_soname\n" ;
    std::string str = ss.str();
    return str ;
}

inline std::string stree::desc_lvid() const
{
    std::stringstream ss ;
    ss << "[stree::desc_lvid\n" ;
    ss << "force_triangulate_lvid.size " << force_triangulate_lvid.size() << "\n" ;

    int sum[3] = {0,0,0} ;

    for(int i=0 ; i < int(soname.size()) ; i++)
    {
        const char* lvn = soname[i].c_str();
        bool starting = false ; // ie MatchAll not MatchStart
        int lvid = find_lvid(lvn, starting);
        bool ift = is_force_triangulate(lvid) ;

        int count_N = count_lvid_nodes(lvid, 'N' );
        int count_R = count_lvid_nodes(lvid, 'R' );
        int count_T = count_lvid_nodes(lvid, 'T' );

        ss
           << " i " << std::setw(4) << i
           << " lvid " << std::setw(4) << lvid
           << " is_force_triangulate " << ( ift ? "YES" : "NO " )
           << " count_N/R/T "
           << std::setw(6) << count_N
           << std::setw(6) << count_R
           << std::setw(6) << count_T
           << " lvn " << lvn
           << "\n"
           ;

        sum[0] += count_N ;
        sum[1] += count_R ;
        sum[2] += count_T ;

    }

    ss << "   " << "    "
       << "      " << "    "
       << "                      " << "   "
       << "   sum_N/R/T "
       << std::setw(6) << sum[0]
       << std::setw(6) << sum[1]
       << std::setw(6) << sum[2]
       << "\n"
       ;

    ss << " nds.size " << nds.size() << "\n" ;
    ss << " rem.size " << rem.size() << "\n" ;
    ss << " tri.size " << tri.size() << "\n" ;
    ss << "]stree::desc_lvid\n" ;
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
    int num_sensor = sensor_name.size() ;
    std::stringstream ss ;
    ss << "stree::desc_sensor" << std::endl
       << " sensor_id.size " << sensor_id.size() << std::endl
       << " sensor_count " << sensor_count << std::endl
       << " sensor_name.size " << num_sensor << std::endl
       ;

    int edgeitems = 20 ;

    ss << "sensor_name[" << std::endl  ;
    for(int i=0; i < num_sensor ; i++)
    {
        if( i < edgeitems || i > num_sensor - edgeitems )
        {
            ss << sensor_name[i].c_str() << std::endl ;
        }
        else if( i == edgeitems )
        {
            ss << "..." << std::endl ;
        }
    }
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

/**
stree::find_lvid
------------------

Find lvid index of solid with name q_soname

**/

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


/**
stree::get_node_vector
-----------------------

The *nds* vector includes all the structural nodes (aka volumes) with the
*rem* and *tri* vectors being subsets of those.

**/

inline const std::vector<snode>* stree::get_node_vector( char _src ) const
{
    const std::vector<snode>* src = nullptr ;
    switch( _src )
    {
        case 'N': src = &nds ; break ;
        case 'R': src = &rem ; break ;
        case 'T': src = &tri ; break ;
    }
    return src ;
}


/**
stree::find_lvid_nodes_
-------------------------

Collect all snode from src vector nds/rem/tri which have the provided lvid shape into nodes vector.

**/

inline void stree::find_lvid_nodes_( std::vector<snode>& nodes, int lvid, char _src ) const
{
    const std::vector<snode>* src = get_node_vector(_src);
    for(unsigned i=0 ; i < src->size() ; i++)
    {
        const snode& sn = (*src)[i] ;
        if( _src == 'N' )
        {
            assert( int(i) == sn.index );
        }
        if(sn.lvid == lvid) nodes.push_back(sn) ;
    }
}

/**
stree::find_lvid_nodes
-----------------------

Collect all snode::index from src vector nds/rem/tri which have the provided lvid shape into nodes vector.
NB this should correpond to the absolute nidx indices not the indices into the selected src (unless the
src is nds which corresponds to all nodes)

**/

inline void stree::find_lvid_nodes( std::vector<int>& nodes, int lvid, char _src ) const
{
    const std::vector<snode>* src = get_node_vector(_src);
    for(unsigned i=0 ; i < src->size() ; i++)
    {
        const snode& sn = (*src)[i] ;
        if( _src == 'N' )
        {
            assert( int(i) == sn.index );
        }
        if(sn.lvid == lvid) nodes.push_back(sn.index) ;
    }
}

/**
stree::find_lvid_nodes
-------------------------

1. lookup int:lvid from q_soname, for starting:true only a string start match is used
2. collect snode::index of all structural snode with int:lvid

**/

inline void stree::find_lvid_nodes( std::vector<int>& nodes, const char* q_soname, bool starting ) const
{
    int lvid = find_lvid(q_soname, starting);
    find_lvid_nodes(nodes, lvid, 'N' );
}


inline int stree::count_lvid_nodes( int lvid, char _src ) const
{
    std::vector<int> nodes ;
    find_lvid_nodes( nodes, lvid, _src );
    return nodes.size();
}



/**
stree::find_lvid_node
---------------------

1. find all lvid node index
2. return the ordinal-th node index

**/
inline int stree::find_lvid_node( const char* q_soname, int ordinal ) const
{
    std::vector<int> nodes ;
    bool starting = true ;
    find_lvid_nodes(nodes, q_soname, starting );
    if(ordinal < 0) ordinal += nodes.size() ; // -ve ordinal counts from back

    return ordinal > -1 && ordinal < int(nodes.size()) ? nodes[ordinal] : -1 ;
}

/**
stree::find_lvid_node
----------------------

1. split q_spec on ':' into vector of elements (str:q_soname, int:middle, int:q_ordinal)

   * default ints are zero
   * HMM: MIDDLE REQUIRED TO BE ZERO CURRENTLY

2. lookup nidx using (str:q_soname,int:ordinal)


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

    bool middle_expect = middle == 0  ;
    if(!middle_expect) std::cerr << "stree::find_lvid_node middle_expect " << std::endl ;
    assert( middle_expect ); // slated for use with global addressing (like MOI)

    int nidx = find_lvid_node(q_soname, ordinal);
    return nidx ;
}

/**
stree::pick_lvid_ordinal_node
-------------------------------

For ridx_type '?' look for the frame first using rem 'R' nodes and then tri 'T' nodes

**/
inline const snode* stree::pick_lvid_ordinal_node( int lvid, int lvid_ordinal, char ridx_type  ) const
{
    const snode* _node = nullptr ;
    assert( ridx_type == 'R' || ridx_type == 'T' || ridx_type == '?' );
    if( ridx_type == 'R' || ridx_type == 'T' )  // remainder OR triangulated
    {
        _node = _pick_lvid_ordinal_node(lvid, lvid_ordinal, ridx_type );
    }
    else if( ridx_type == '?' )
    {
        _node = _pick_lvid_ordinal_node(lvid, lvid_ordinal, 'R' );
        if(_node == nullptr)
        {
            _node = _pick_lvid_ordinal_node(lvid, lvid_ordinal, 'T' );
        }
    }
    return _node ;
}


/**
stree::_pick_lvid_ordinal_node
-------------------------------

Returns selected node (pointer into nds vector)

1. collect indices of all snode (volumes) with lvid shape from the ridx_type vector nds/rem/tri
2. select the ordinal-th snode (-ve ordinal counts from back of the selected set)
3. use selected nn node index from find_lvid_nodes, which is an absolute nidx
   to return pointer into the nds vector or nullptr

**/
inline const snode* stree::_pick_lvid_ordinal_node( int lvid, int lvid_ordinal, char ridx_type  ) const
{
    std::vector<int> nn ;
    find_lvid_nodes( nn, lvid, ridx_type );

    int num = nn.size() ;
    if( lvid_ordinal < 0 ) lvid_ordinal += num ;
    bool valid =  lvid_ordinal > -1 && lvid_ordinal < num ;
    int nidx = valid ? nn[lvid_ordinal] : -1 ;

    const snode* nds_data = nds.data() ;
    return nidx > -1 && nidx < int(nds.size()) ? nds_data + nidx : nullptr ;
}

/**
stree::pick_lvid_ordinal_repeat_ordinal_inst_
----------------------------------------------

1. pick_lvid_ordinal_node provides the snode, giving the repeat_index (aka gas_idx)
2. find_inst_gas uses that repeat_index and the repeat_ordinal to access the instance index

This is trying to repeat the MOI logic from CSGTarget::getInstanceTransform

**/

inline int stree::pick_lvid_ordinal_repeat_ordinal_inst_( int lvid, int lvid_ordinal, int repeat_ordinal ) const
{
    const snode* n = _pick_lvid_ordinal_node(lvid, lvid_ordinal, 'N' );
    if( n == nullptr ) return -1 ;

    int q_gas_idx = n->repeat_index ;  // aka gas_idx
    int q_gas_ordinal = repeat_ordinal ;
    int inst_idx = find_inst_gas( q_gas_idx, q_gas_ordinal );

    return inst_idx ;
}


/**
stree::parse_spec
------------------

Parse string of below form into lvid index by name lookup
from 1st field and integers lvid_ordinal repeat_ordinal from 2nd and 3rd::

    Hama:0:0
    NNVT:0:0
    Hama:0:1000
    NNVT:0:1000
    sDeadWater:0:0
    GZ1.A06_07_FlangeI_Web_FlangeII:0:0
    GZ1.B06_07_FlangeI_Web_FlangeII:0:0
    GZ1.A06_07_FlangeI_Web_FlangeII:15:0
    GZ1.B06_07_FlangeI_Web_FlangeII:15:0
    0:0:0

When no 2nd and 3rd field is provided eg with "sDeadWater" the
ordinals default to 0.

A integer string in the first field is converted to lvid int.

TODO: get this to ignore comments in the q_spec line like::

    sDeadWater:0:-1   # some comment

**/

inline int stree::parse_spec(
    int& lvid,
    int& lvid_ordinal,
    int& repeat_ordinal,
    const char* q_spec ) const
{
    std::vector<std::string> elem ;
    sstr::Split(q_spec, ':', elem );

    const char* q_soname  = elem.size() > 0 ? elem[0].c_str() : nullptr ;
    const char* q_lvid_ordinal  = elem.size() > 1 ? elem[1].c_str() : nullptr ;
    const char* q_repeat_ordinal = elem.size() > 2 ? elem[2].c_str() : nullptr ;


    if(sstr::IsInteger(q_soname))
    {
        lvid = sstr::To<int>(q_soname) ;
    }
    else
    {
        bool starting = true ;
        lvid = find_lvid(q_soname, starting);
    }

    if(lvid == -1 )
    {
        std::cerr << "stree::parse_spec FAILED to find lvid for q_soname [" << ( q_soname ? q_soname : "-" ) << "]\n" ;
        std::cerr << desc_soname() ;
    }
    if( lvid == -1 ) return -1 ;

    lvid_ordinal = q_lvid_ordinal  ? std::atoi(q_lvid_ordinal) : 0 ;
    repeat_ordinal = q_repeat_ordinal ? std::atoi(q_repeat_ordinal)  : 0 ;

    return 0 ;
}

inline int stree::pick_lvid_ordinal_repeat_ordinal_inst( const char* q_spec ) const
{
    int lvid ;
    int lvid_ordinal ;
    int repeat_ordinal ;
    [[maybe_unused]] int rc = parse_spec( lvid, lvid_ordinal, repeat_ordinal, q_spec );
    assert( rc == 0 );
    int inst_idx = pick_lvid_ordinal_repeat_ordinal_inst_( lvid, lvid_ordinal, repeat_ordinal );
    return inst_idx ;
}

/**
stree::get_frame_f4
--------------------

transitional method to match with CSGFoundry::getFrame

See ~/o/notes/issues/sframe_dtor_double_free_from_CSGOptiX__initFrame.rst

**/

inline void stree::get_frame_f4( sframe& fr, int idx ) const
{
    typedef glm::tmat4x4<float> M44 ;

    const M44* _m2w = get_inst_f4(idx);
    const M44* _w2m = get_iinst_f4(idx);

    assert( sizeof(M44) == sizeof(fr.m2w ) );
    memcpy( fr.m2w.data(), _m2w , sizeof(M44) );
    memcpy( fr.w2m.data(), _w2m , sizeof(M44) );
}


/**
stree::get_frame_moi
---------------------

Special cased MOI envvar starting "EXTENT:" normally MOI is of the below form::

    sWaterTube:0:-1


**/

inline sfr stree::get_frame_moi() const
{
    float _extent = sstr::StartsWith(MOI, _EXTENT_PFX) ? sstr::To<float>( MOI + strlen(_EXTENT_PFX) ) : 0.f ;
    sfr mf =  _extent > 0.f ? sfr::MakeFromExtent<float>(_extent) : get_frame(MOI) ;
    return mf ;
}


/**
stree::get_frame
------------------

1. parse_spec from q_spec get (lvid, lvid_ordinal, repeat_ordinal)

Q: An instance may encompasses multiple lv (and multiple snode)
   so which nidx is collected together with the inst
   transforms into inst_nidx ? The outer one would be most useful.

A: By observation the outer instance node is collected into inst_nidx

**/

inline sfr stree::get_frame(const char* q_spec ) const
{
    int lvid ;
    int lvid_ordinal ;
    int repeat_ordinal ;
    int parse_rc = parse_spec( lvid, lvid_ordinal, repeat_ordinal, q_spec );

    if(parse_rc != 0) std::cerr
        << "stree::get_frame"
        << " FATAL parse_spec failed "
        << " q_spec [" << ( q_spec ? q_spec : "-" ) << "]"
        << " parse_rc " << parse_rc
        << "\n"
        ;
    assert( parse_rc == 0 );

    sfr f ;
    f.set_name(q_spec);


    [[maybe_unused]] int get_rc = 0 ;
    if( repeat_ordinal == -1 || repeat_ordinal == -2 || repeat_ordinal == -3 )
    {
        get_rc = get_frame_global(  f,  lvid, lvid_ordinal, repeat_ordinal );
    }
    else
    {
        get_rc = get_frame_instanced(f,  lvid, lvid_ordinal, repeat_ordinal );
    }

    if(get_rc != 0 ) std::cerr
        << "stree::get_frame FAIL q_spec[" << ( q_spec ? q_spec : "-" ) << "]\n"
        << " THIS CAN BE CAUSED BY NOT USING REPEAT_ORDINAL -1 (LAST OF TRIPLET) FOR GLOBAL GEOMETRY "
        << "\n"
        ;

    assert( get_rc == 0 );
    return f ;
}


/**
stree::has_frame
------------------

The spec are of form::

    Hama:0:1000
    solidXJanchor:20:-1
    sSurftube_38V1_0:0:-1

Where the three fields provide the ints::

    (lvid, lvid_ordinal, repeat_ordinal)


From v0.3.7
   returns false when q_spec is invalid (formerly asserted),
   Invalid q_spec is usually because the lv name starting the q_spec
   is not present in the geometry

**/

inline bool stree::has_frame(const char* q_spec) const
{
    int lvid ;
    int lvid_ordinal ;
    int repeat_ordinal ;
    int parse_rc = parse_spec( lvid, lvid_ordinal, repeat_ordinal, q_spec );
    //assert( parse_rc == 0 );
    if(parse_rc != 0)
    {
        std::cerr
            << "stree::has_frame"
            << " FATAL parse_spec failed "
            << " q_spec [" << ( q_spec ? q_spec : "-" ) << "]"
            << " parse_rc " << parse_rc
            << "\n"
            ;
        return false ;
    }

    sfr f ;
    f.set_name(q_spec);

    int get_rc = 0 ;
    if( repeat_ordinal == -1 || repeat_ordinal == -2 || repeat_ordinal == -3)
    {
        get_rc = get_frame_global(  f,  lvid, lvid_ordinal, repeat_ordinal );
    }
    else
    {
        get_rc = get_frame_instanced(f,  lvid, lvid_ordinal, repeat_ordinal );
    }

    if(get_rc != 0 ) std::cerr
        << "stree::has_frame FAIL q_spec[" << ( q_spec ? q_spec : "-" ) << "]\n"
        << " THIS CAN BE CAUSED BY NOT USING REPEAT_ORDINAL -1 (LAST OF TRIPLET) FOR GLOBAL GEOMETRY "
        << "\n"
        ;

    return get_rc == 0 ;
}



/**
stree::get_frame_instanced
----------------------------

1. pick_lvid_ordinal_repeat_ordinal_inst_ gives ii from (lvid, lvid_ordinal, repeat_ordinal)

   * for non-instanced *ii* "instance-index" will be zero

2. get_inst, get_iinst : lookup the instance transforms using *ii*

   * for non-instanced will yield identity transforms

3.  use inst_nidx and nds to get the snode for the ii

   * for global that will be the outer world volume

4. get_prim_aabb : find bounding box of the snode by delving into
   the transformed CSG constituent nds for the node.lvid


**/

inline int stree::get_frame_instanced(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const
{
    int ii = pick_lvid_ordinal_repeat_ordinal_inst_( lvid, lvid_ordinal, repeat_ordinal );

    const glm::tmat4x4<double>* m2w = get_inst(ii);
    const glm::tmat4x4<double>* w2m = get_iinst(ii);

    bool missing_transform = !m2w || !w2m ;

    if(missing_transform) std::cerr
        << "stree::get_frame_instanced FAIL missing_transform "
        << " lvid " << lvid
        << " lvid_ordinal " << lvid_ordinal
        << " repeat_ordinal " << repeat_ordinal
        << " w2m " << ( w2m ? "YES" : "NO " )
        << " m2w " << ( m2w ? "YES" : "NO " )
        << " ii " << ii
        << "\n"
        ;

    if(missing_transform) return 1 ;
    assert( m2w );
    assert( w2m );

    int nidx = inst_nidx[ii] ;
    const snode& nd = nds[nidx] ;

    std::array<double,6> bb ;
    get_prim_aabb( bb.data(), nd, nullptr );

    if(get_frame_dump) std::cout
        << "stree::get_frame_instanced"
        << "\n"
        << " lvid " << lvid
        << " soname[lvid] " << soname[lvid]
        << " soname[nd.lvid] " << soname[nd.lvid]
        << "\n"
        << " lvid_ordinal " << lvid_ordinal
        << " repeat_ordinal " << repeat_ordinal
        << " ii " << ii
        << " nidx " << nidx
        << "\n"
        << " nd.desc " << nd.desc()
        << "\n"
        << " bb \n"
        << s_bb::Desc( bb.data() )
        << "\n"
        ;

    //assert( nd.lvid == lvid );
    // lvid will not in general match
    // because there are multiple lv within the instance
    // and the access technique goes via the gas_idx ?

    //assert( nd.repeat_ordinal == repeat_ordinal );
    // not so for globals

    // TODO: aux0/1/2 arrange layout of integers

    s_bb::CenterExtent( f.ce_data(), bb.data() );

    f.m2w = *m2w ;
    f.w2m = *w2m ;

    return 0 ;
}



/**
stree::get_frame_remainder
--------------------------------

Note the similartity to MOI targetting,
at CSG level which is handled by::

    CSGTarget::getFrameComponents
    CSGTarget::getGlobalCenterExtent
    CSGImport::importSolidRemainder

The info is definitely here, just have different access
here at stree level.

* look inside rem snode for the specified (lvid, lvid_ordinal) ?

TODO: avoid the duplication in frame access impls

**/
inline int stree::get_frame_remainder(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const
{
    return _get_frame_global( f, lvid, lvid_ordinal, repeat_ordinal, 'R' );
}
inline int stree::get_frame_triangulate(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const
{
    return _get_frame_global( f, lvid, lvid_ordinal, repeat_ordinal, 'T' );
}
inline int stree::get_frame_global(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const
{
    return _get_frame_global( f, lvid, lvid_ordinal, repeat_ordinal, '?' );
}


/**
stree::_get_frame_global
--------------------------

This is called for special cased -ve repeat_ordinal, which
is only appropriate for global non-instanced volumes.

1. find the snode using (lvid, lvid_ordinal, ridx_type)
2. compute bounding box and hence center_extent for the snode
3. form frame transforms m2w/w2m using SCenterExtentFrame or not
   depending on repeat_ordinal -1/-2/-3

Global repeat_ordinal special case convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

repeat_ordinal:-1
   sets CE only, does not set m2w w2m into the frame
   [WHAT USE IS THIS ?]

repeat_ordinal:-2
   sets CE, m2w, w2m into the frame using SCenterExtentFrame with rtp_tangential:false

repeat_ordinal:-3
   sets CE, m2w, w2m into the frame using SCenterExtentFrame with rtp_tangential:true


27 May 2025 behaviour change for repeat_ordinal:-1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WIP: test this

Formerly the stree::_get_frame_global repeat_ordinal:-1 gave frames
with transforms that CSGTarget::getFrameComponents
would need repeat_ordinal:-2 for.

The stree::_get_frame_global implementation is
now aligned with CSGTarget::getFrameComponents
to avoid the need to keep swapping MOI -1/-2 arising from
a former difference in the convention used.

**/

inline int stree::_get_frame_global(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal, char ridx_type ) const
{
    assert( repeat_ordinal == -1 || repeat_ordinal == -2 || repeat_ordinal == -3 );
    const snode* _node = pick_lvid_ordinal_node( lvid, lvid_ordinal, ridx_type );
    if(_node == nullptr) return 1 ;

    const snode& node = *_node ;

    std::array<double,4> ce = {} ;
    std::array<double,6> bb = {} ;
    int rc = get_node_ce_bb( ce, bb, node );
    f.set_ce(ce.data() );

    if( repeat_ordinal == -2 || repeat_ordinal == -3 )
    {
        bool rtp_tangential = repeat_ordinal == -3 ? true : false ;
        bool extent_scale = false ;
        SCenterExtentFrame<double> cef(ce[0], ce[1], ce[2], ce[3], rtp_tangential, extent_scale ) ;
        f.m2w = cef.model2world ;
        f.w2m = cef.world2model ;
    }

    if(get_frame_dump) std::cout
        << "stree::get_frame_remainder"
        << "\n"
        << " lvid " << lvid
        << " soname[lvid] " << soname[lvid]
        << " soname[node.lvid] " << ( soname[node.lvid] )
        << "\n"
        << " lvid_ordinal " << lvid_ordinal
        << " repeat_ordinal " << repeat_ordinal
        << "\n"
        << " node.desc " << ( node.desc())
        << "\n"
        << " bb \n"
        << s_bb::Desc( bb.data() )
        << "\n"
        ;

    return rc ;
}

inline int stree::get_node_ce_bb(    std::array<double,4>& ce , std::array<double,6>& bb,  const snode& node ) const
{
    int rc = get_node_bb(bb, node);
    s_bb::CenterExtent( ce.data(), bb.data() );
    return rc ;
}

inline int stree::get_node_bb(  std::array<double,6>& bb , const snode& node ) const
{
    int lvid = node.lvid ;

    std::vector<const sn*> bds ;         // binary tree nodes
    sn::GetLVNodesComplete(bds, lvid);   // many nullptr in unbalanced deep complete binary trees
    int bn = bds.size();                 // number of binary tree nodes

    std::vector<const sn*> lns ;
    sn::GetLVListnodes( lns, lvid );
    //int num_sub_total = sn::GetChildTotal( lns );

    int ln = lns.size();
    assert( ln == 0 || ln == 1 ); // simplify initial impl  : see CSGImport::importPrim

    std::ostream* out = nullptr ;

    std::vector<const sn*> subs ;

    for(int i=0 ; i < bn ; i++)
    {
        const sn* n = bds[i];
        int  typecode = n ? n->typecode : CSG_ZERO ;

        if(n && n->is_listnode())
        {
            // hmm subtracted holes will no contribute to bbox
            int num_sub = n->child.size() ;
            for(int j=0 ; j < num_sub ; j++)
            {
                const sn* c = n->child[j];
                subs.push_back(c);
            }
        }
        else
        {
            bool leaf = CSG::IsLeaf(typecode) ;

            if(0) std::cout
                << "stree::get_frame_remainder"
                << " i " << std::setw(2) << i
                << " typecode " << typecode
                << " leaf " << ( leaf ? "Y" : "N" )
                << "\n"
                ;

            std::array<double,6> n_bb ;
            double* n_aabb = leaf ? n_bb.data() : nullptr ;
            const Tran<double>* tv = leaf ? get_combined_tran_and_aabb( n_aabb, node, n, nullptr ) : nullptr ;

            if(tv && leaf && n_aabb && !n->is_complement_primitive()) s_bb::IncludeAABB( bb.data(), n_aabb, out );
        }
    }


    // NOT FULLY TESTED : but it succeeds to do nothing with subtracted multiunion of holes (that becomes listnode)
    int num_sub_total = subs.size();
    for( int i=0 ; i < num_sub_total ; i++ )
    {
        const sn* n = subs[i];
        bool leaf = CSG::IsLeaf(n->typecode) ;
        assert(leaf);

        std::array<double,6> n_bb ;
        double* n_aabb = leaf ? n_bb.data() : nullptr ;
        const Tran<double>* tv = leaf ? get_combined_tran_and_aabb( n_aabb, node, n, nullptr ) : nullptr ;

        if(tv && leaf && n_aabb && !n->is_complement_primitive()) s_bb::IncludeAABB( bb.data(), n_aabb, out );
        // HMM does the complement message get thru to listnode subs ?
    }
    return 0 ;
}









/**
stree::get_sub_sonames
-----------------------

// TODO: should this be using sfactor ?

**/
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


/**
stree::Name
------------

HMM: tail stripping now done at collection with sstr::StripTail_Unique

**/
inline std::string stree::Name( const std::string& name, bool strip ) // static
{
    return strip ? sstr::StripTail(name, "0x") : name ;
}
inline std::string stree::get_lvid_soname(int lvid, bool strip) const
{
    if(lvid < 0 || lvid >= int(soname.size())) return "bad_lvid" ;
    return Name(soname[lvid], strip) ;
}

inline const std::string& stree::get_lvid_soname_(int lvid) const
{
    assert( lvid >= 0 && lvid < int(soname.size()) ) ;
    return soname[lvid] ;
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

    if(level > 1) std::cout
        << "stree::get_mmlabel"
        << " level " << level
        << " num_ridx " << num_ridx
        << "\n"
        ;

    for(int ridx=0 ; ridx < num_ridx ; ridx++)
    {
        int num_prim = get_ridx_subtree(ridx) ;
        int olvid    = get_ridx_olvid(ridx) ;

        assert( olvid < int(soname.size()) );
        std::string name = get_lvid_soname(olvid, true);

        std::stringstream ss ;
        ss << num_prim << ":" << name ;
        std::string mmlabel = ss.str();

        if(level > 1) std::cout
            << "stree::get_mmlabel"
            << " level " << level
            << " ridx " << ridx
            << " mmlabel " << mmlabel
            << "\n"
            ;

        names.push_back(mmlabel);
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

Used by::

   stree::get_ancestors
   stree::get_node_product

An outer node is either the root node which has no parent
or some other node with a parent from a different repeat_index.
Typically the different repeat_index with be the global zero.

The outer nodes correspond to base nodes of the instances,
and similarly the root node is the base node of the remainder nodes.

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
    For *nidx* within instanced nodes this will only include ancestor
    nodes within that same instance. Note also that the outer node of
    the instance is BY DESIGN : NOT INCLUDED.

**/

inline void stree::get_ancestors(
    std::vector<int>& ancestors,
    int nidx,
    bool local,
    std::ostream* out ) const
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


/**
stree::get_node_transform
---------------------------

Returns local (relative to parent) transforms for the nidx snode

**/

inline void stree::get_node_transform( glm::tmat4x4<double>& m2w_, glm::tmat4x4<double>& w2m_, int nidx ) const
{
    assert( w2m.size() == m2w.size() );
    assert( nidx > -1 && nidx < int(m2w.size()));

    m2w_ = m2w[nidx];
    w2m_ = w2m[nidx];
}

/**
stree::get_node_product
-------------------------

local:true
   note that the get_ancestors does not include the outer node index,
   where the outer node is the one that has parent of different repeat_idx,
   or root with no parent.

   The reason to skip the outer node is because the transform for that node
   will differ for all instances whereas with local:true are operating
   within the frame of the instance such that the transform product will
   be the same for all instances.  Indeed that skipped transform
   will become part of the instance transforms.


Q: What transforms are provided when called from the nidx of outer instanced nodes ?
A: In that case num_nodes=0 so identity transforms are returned.

**/

inline void stree::get_node_product(
                      glm::tmat4x4<double>& m2w_,
                      glm::tmat4x4<double>& w2m_,
                      int nidx,
                      bool local,
                      bool reverse,
                      std::ostream* out ) const
{
    std::vector<int> nodes ;
    get_ancestors(nodes, nidx, local, out);  // root-first-order (from collecting parent links then reversing the vector)

    bool is_local_outer = local && is_outer_node(nidx) ;
    if(is_local_outer == false ) nodes.push_back(nidx);
    // dont include the local_outer node, here either


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

NB::

     local = node.repeat_index > 0

**The local argument to get_node_product drastically changes the
character of the returned transforms for the global ridx:0 and
the "local" ridx>0 instances by changing which transforms
are included in the product**

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


local:true for instances node.repeat_index > 0
    get_node_product with local:true gives structural transform
    product within the instance, excluding the outer node transform.
    That transform product will often be identity.


**/

inline void stree::get_combined_transform(
    glm::tmat4x4<double>& t,
    glm::tmat4x4<double>& v,
    const snode& node,
    const sn* nd,
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
        sn::NodeTransformProduct(nd->idx(), tc, vc, false, out );  // reverse:false
    }

    // combine structural (volume level) and CSG (solid level) transforms
    t = tt * tc ;
    v = vc * vv ;

    if(out) *out << stra<double>::Desc( t, v, "(tt*tc)", "(vc*vv)" ) << std::endl << std::endl ;
}

inline std::string stree::desc_combined_transform(
    glm::tmat4x4<double>& t,
    glm::tmat4x4<double>& v,
    const snode& node,
    const sn* nd ) const
{
    std::stringstream ss ;
    ss << "stree::desc_combined_transform" << std::endl;
    get_combined_transform(t, v, node, nd, &ss );
    std::string str = ss.str();
    return str ;
}

/**
stree::get_combined_tran_and_aabb
--------------------------------------

Critical usage of this from CSGImport::importNode

0. early exits returning nullptr for non leaf nodes
1. gets combined structural(snode.h) and CSG tree(sn.h) transform
2. collects that combined transform and its inverse (t,v) into Tran instance
3. copies leaf frame bbox values from the CSG nd into callers aabb array
4. transforms the bbox of the callers aabb array using the combined structural node
   + tree node transform


Note that sn::uncoincide needs CSG tree frame AABB but whereas this needs leaf
frame AABB. These two demands are met by changing the AABB frame
within sn::postconvert

**/

inline const Tran<double>* stree::get_combined_tran_and_aabb(
    double* aabb,
    const snode& node,
    const sn* nd,
    std::ostream* out
    ) const
{
    assert( nd );
    if(!CSG::IsLeaf(nd->typecode)) return nullptr ;

    glm::tmat4x4<double> t(1.) ;
    glm::tmat4x4<double> v(1.) ;
    get_combined_transform(t, v, node, nd, out );

    // NB ridx:0 full stack of transforms from root down to CSG constituent nodes
    //    ridx>0 only within the instance and within constituent CSG tree

    const Tran<double>* tv = new Tran<double>(t, v);

    nd->copyBB_data( aabb );
    stra<double>::Transform_AABB_Inplace(aabb, t);

    return tv ;
}


/**
stree::get_transformed_aabb
----------------------------

snode.repeat_index:0
    full stack of transforms from root down into CSG constituent sn nodes
snode.repeat_index>0
    only within the instance and down into constituent sn nodes

**/

inline void stree::get_transformed_aabb(
    double* aabb,
    const snode& node,
    const sn* nd,
    std::ostream* out
    ) const
{
    assert( nd );
    if(!CSG::IsLeaf(nd->typecode)) return ;

    glm::tmat4x4<double> t(1.) ;
    glm::tmat4x4<double> v(1.) ;
    get_combined_transform(t, v, node, nd, out );

    nd->copyBB_data( aabb );
    stra<double>::Transform_AABB_Inplace(aabb, t);
}


/**
stree::get_prim_aabb
---------------------

Follow pattern of::

    CSGImport::importPrim_
    CSGImport::importNode

1. gets CSG constituent nds for the node.lvid with sn::GetLVNodesComplete
2. combines the transformed constituent bounding box

HMM: THIS DOES NOT CONSIDER LISTNODE

**/
inline void stree::get_prim_aabb( double* aabb, const snode& node, std::ostream* out ) const
{
    std::vector<const sn*> nds ;
    sn::GetLVNodesComplete(nds, node.lvid); // many nullptr in unbalanced deep complete binary trees
    int numParts = nds.size();

    std::array<double,6> pbb = {} ;

    for(int i=0 ; i < numParts ; i++)
    {
        int partIdx = i ;
        const sn* nd = nds[partIdx];

        int  typecode = nd ? nd->typecode : CSG_ZERO ;
        bool leaf = CSG::IsLeaf(typecode) ;
        if(!leaf) continue ;

        bool is_complemented_primitive = nd->complement && CSG::IsPrimitive(typecode) ;
        if(is_complemented_primitive) continue ;

        bool external_bbox_is_expected = CSG::ExpectExternalBBox(typecode);
        bool expect = external_bbox_is_expected == false ;
        if(!expect) std::cerr << " NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED : DEFERRED UNTIL HAVE EXAMPLES\n" ;
        assert(expect);
        if(!expect) std::raise(SIGINT);

        std::array<double,6> nbb ;
        get_transformed_aabb( nbb.data(), node, nd, out );
        s_bb::IncludeAABB( pbb.data(), nbb.data(), out );
    }
    for(int i=0 ; i < 6 ; i++) aabb[i] = pbb[i] ;
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



/**
stree::desc_node_solids
-------------------------

HUH: this is horribly repetitive and long presentation of soname for every node

**/


inline std::string stree::desc_node_solids() const
{
    int num_nodes = get_num_nodes();
    std::stringstream ss ;
    ss << "stree::desc_node_solids num_nodes " << num_nodes  << std::endl ;
    for(int nidx=0 ; nidx < num_nodes ; nidx++)
    {
        ss
            << " nidx " << std::setw(6) << nidx
            << " so " << get_soname(nidx)
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}



/**
stree::desc_solids
-------------------

OBSERVE THAT THE stree::solids ARE NOT PERSISTED, INSTEAD USE sn::Get methods
to access the s_csg.h persisted sn.h

**/

inline std::string stree::desc_solids() const
{
    int num_solids = solids.size() ;
    std::stringstream ss ;
    ss << "stree::desc_solids num_solids " << num_solids  << std::endl ;
    for(int i=0 ; i < num_solids ; i++)
    {
        const sn* root = solids[i] ;
        ss
            << " (sn)root.lvid " << std::setw(3) << root->lvid
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}


inline std::string stree::desc_solid(int lvid) const
{
    const sn* root = sn::GetLVRoot(lvid) ;
    const std::string& lvn = get_lvid_soname_(lvid) ;
    assert( root ) ;

    std::stringstream ss ;
    ss << "stree::desc_solid"
       << " lvid " << lvid
       << " lvn " << lvn
       << " root " << ( root ? "Y" : "N" )
       << " " << ( root ? root->rbrief() : "" )
       ;
    std::string str = ss.str();
    return str ;
}



/**
stree::make_trs
-----------------

This is used from U4Tree::simtrace_scan as the basis for u4/tests/U4SimtraceTest.sh
 HMM: this is based on GTD: "GGeo Transform Debug" so it is not future safe

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
    NP* _tri = NPX::ArrayFromVec<int,snode>( tri, snode::NV ) ;
    NP* _m2w = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( m2w, 4, 4 ) ;
    NP* _w2m = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( w2m, 4, 4 ) ;
    NP* _gtd = NPX::ArrayFromVec<double,glm::tmat4x4<double>>( gtd, 4, 4 ) ;

    fold->add( NDS, _nds );
    fold->add( REM, _rem );
    fold->add( TRI, _tri );
    fold->add( M2W, _m2w );
    fold->add( W2M, _w2m );
    fold->add( GTD, _gtd );

    NP* _mtname = NPX::Holder(mtname) ;
    NP* _mtname_no_rindex = NPX::Holder(mtname_no_rindex) ;
    NP* _mtindex = NPX::ArrayFromVec<int,int>( mtindex );
    NP* _mtline = NPX::ArrayFromVec<int,int>( mtline );
    NP* _force_triangulate_lvid = NPX::ArrayFromVec<int,int>( force_triangulate_lvid );


    fold->add( MTNAME,  _mtname );
    fold->add( MTNAME_NO_RINDEX,  _mtname_no_rindex );
    fold->add( MTINDEX, _mtindex );
    fold->add( MTLINE , _mtline );

    fold->add( FORCE_TRIANGULATE_LVID, _force_triangulate_lvid ) ;

    if(material) fold->add_subfold( MATERIAL, material );
    if(surface)  fold->add_subfold( SURFACE,  surface );
    if(mesh)     fold->add_subfold( MESH,     mesh );


    fold->add( SUNAME,   NPX::Holder(suname) );
    fold->add( IMPLICIT, NPX::Holder(implicit) );
    //fold->add( SUINDEX,  NPX::ArrayFromVec<int,int>( suindex )  );

    NPFold* f_standard = standard->serialize() ;
    fold->add_subfold( STANDARD, f_standard );
    fold->add_subfold( _CSG, _csg->serialize() );
    fold->add( SONAME, NPX::Holder(soname) );

    fold->add( DIGS, NPX::Holder(digs) );
    fold->add( SUBS, NPX::Holder(subs) );

    NPFold* f_subs_freq = subs_freq->serialize() ;
    fold->add_subfold( SUBS_FREQ, f_subs_freq );

    NP* _factor = NPX::ArrayFromVec<int,sfactor>( factor, sfactor::NV );

    NP* _inst = NPX::ArrayFromVec<double, glm::tmat4x4<double>>( inst, 4, 4) ;
    NP* _iinst = NPX::ArrayFromVec<double, glm::tmat4x4<double>>( iinst, 4, 4) ;

    // inst_f4 crucially used from CSGImport::importInst
    NP* _inst_f4 = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( inst_f4, 4, 4) ;
    NP* _iinst_f4 = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( iinst_f4, 4, 4) ;

    NP* _inst_info = NPX::ArrayFromVec<int,int4>( inst_info, 4 ) ;
    NP* _inst_nidx = NPX::ArrayFromVec<int,int>( inst_nidx ) ;
    NP* _sensor_id = NPX::ArrayFromVec<int,int>( sensor_id ) ;
    NP* _sensor_name = NPX::Holder(sensor_name);
    NP* _mtindex_to_mtline = NPX::ArrayFromDiscoMap<int>( mtindex_to_mtline ) ;


    if(ssys::getenvbool(stree__populate_prim_nidx))
    {
        NP* _prim_nidx = NPX::ArrayFromVec<int,int>( prim_nidx ) ;
        fold->add( PRIM_NIDX, _prim_nidx );
    }
    else if(ssys::getenvbool(stree__populate_nidx_prim))
    {
        NP* _prim_nidx = NPX::ArrayFromVec<int,int>( prim_nidx ) ;
        NP* _nidx_prim = NPX::ArrayFromVec<int,int>( nidx_prim ) ;
        fold->add( PRIM_NIDX, _prim_nidx );
        fold->add( NIDX_PRIM, _nidx_prim );
    }


    fold->add( FACTOR, _factor );
    fold->add( INST,   _inst );
    fold->add( IINST,   _iinst );
    fold->add( INST_F4,   _inst_f4 );
    fold->add( IINST_F4,   _iinst_f4 );
    fold->add( INST_INFO,   _inst_info );
    fold->add( INST_NIDX,   _inst_nidx );
    fold->add( SENSOR_ID,   _sensor_id );
    fold->add( SENSOR_NAME, _sensor_name );
    fold->add( MTINDEX_TO_MTLINE, _mtindex_to_mtline  );



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
    ImportArray<snode, int>( tri,                  fold->get(TRI) );
    ImportArray<glm::tmat4x4<double>, double>(m2w, fold->get(M2W) );
    ImportArray<glm::tmat4x4<double>, double>(w2m, fold->get(W2M) );
    ImportArray<glm::tmat4x4<double>, double>(gtd, fold->get(GTD) );

    ImportNames( soname,            fold->get(SONAME) );
    ImportNames( mtname,            fold->get(MTNAME) );
    ImportNames( mtname_no_rindex,  fold->get(MTNAME_NO_RINDEX) );
    ImportNames( suname,            fold->get(SUNAME) );
    ImportNames( implicit,          fold->get(IMPLICIT) );

    ImportArray<int, int>( force_triangulate_lvid, fold->get(FORCE_TRIANGULATE_LVID) );

    ImportArray<int, int>( mtindex, fold->get(MTINDEX) );
    //ImportArray<int, int>( suindex, fold->get(SUINDEX) );

    NPX::DiscoMapFromArray<int>( mtindex_to_mtline, fold->get(MTINDEX_TO_MTLINE) );

    NPFold* f_standard = fold->get_subfold(STANDARD) ;

    if(f_standard->is_empty())
    {
        std::cerr
            << "stree::import skip asserts for empty f_standard : assuming trivial test geometry "
            << std::endl
            ;
    }
    else
    {
        standard->import(f_standard);

        assert( standard->bd );
        NPX::VecFromArray<int4>( vbd, standard->bd );
        standard->bd->get_names( bdname );

        assert( standard->bnd );
        //import_bnd( standard->bnd );
    }


    NPFold* csg_f = fold->get_subfold(_CSG) ;

    if(csg_f == nullptr) std::cerr
        << "stree::import"
        << " FAILED : DUE TO LACK OF subfold _CSG : " << _CSG
        << std::endl
        ;

    _csg->import(csg_f);


    ImportNames( digs, fold->get(DIGS) );
    ImportNames( subs, fold->get(SUBS) );

    NPFold* f_subs_freq = fold->get_subfold(SUBS_FREQ) ;
    subs_freq->import(f_subs_freq);

    ImportArray<sfactor, int>( factor, fold->get(FACTOR) );

    material = fold->get_subfold( MATERIAL) ;
    surface  = fold->get_subfold( SURFACE ) ;
    mesh     = fold->get_subfold( MESH ) ;

    ImportArray<glm::tmat4x4<double>, double>(inst,   fold->get(INST));
    ImportArray<glm::tmat4x4<double>, double>(iinst,  fold->get(IINST));
    ImportArray<glm::tmat4x4<float>, float>(inst_f4,  fold->get(INST_F4));
    ImportArray<glm::tmat4x4<float>, float>(iinst_f4, fold->get(IINST_F4));

    ImportArray<int, int>( sensor_id, fold->get(SENSOR_ID) );
    sensor_count = sensor_id.size();
    ImportNames( sensor_name, fold->get(SENSOR_NAME) );

    ImportArray<int4,int>( inst_info, fold->get(INST_INFO) );
    ImportArray<int, int>( inst_nidx, fold->get(INST_NIDX) );


    if(ssys::getenvbool(stree__populate_prim_nidx))
    {
        ImportArray<int, int>( prim_nidx, fold->get(PRIM_NIDX) );
    }
    else if(ssys::getenvbool(stree__populate_nidx_prim))
    {
        ImportArray<int, int>( prim_nidx, fold->get(PRIM_NIDX) );
        ImportArray<int, int>( nidx_prim, fold->get(NIDX_PRIM) );
    }

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
stree::FindForceTriangulateLVID
--------------------------------

Canonical usage from stree::findForceTriangulateLVID

1. if _force_triangulate_solid is nullptr return doing nothing
2. split _force_triangulate_solid delimited string into *force* vector of potential solid names

   * when have lots of solid names to force triangulate its easier to manage these in a file, see stree::findForceTriangulateLVID
   * when the string contains '\n' sstr::SplitTrimSuppress overrides delim to '\n'

3. for each solid name found in the *_soname* vector collect corresponding indices into *lvid* vector

**/


inline void stree::FindForceTriangulateLVID(std::vector<int>& lvid, const std::vector<std::string>& _soname, const char* _force_triangulate_solid, char delim  )  // static
{
    if(_force_triangulate_solid == nullptr) return ;

    std::vector<std::string> force ;
    sstr::SplitTrimSuppress( _force_triangulate_solid, delim, force );
    unsigned num_force = force.size();

    for(unsigned i=0 ; i < num_force ; i++)
    {
        const char* f = force[i].c_str() ;
        int lv = slist::FindIndex(_soname, f );
        if(lv == -1) std::cerr << "stree::FindForceTriangulateLVID name not found [" << ( f ? f : "-" ) << "]\n" ;
        if(lv > -1) lvid.push_back(lv);
    }
}



inline std::string stree::descForceTriangulateLVID() const
{
    std::stringstream ss ;
    ss << "stree::descForceTriangulateLVID\n"
       << " force_triangulate_solid [" << ( force_triangulate_solid ? force_triangulate_solid : "-" ) << "]\n"
       << " force_triangulate_lvid.size  " << force_triangulate_lvid.size() << "\n"
       << " soname.size  " << soname.size() << "\n"
       ;
    std::string str = ss.str();
    return str ;
}


/**
stree::is_force_triangulate
----------------------------

How this info gets used is spread over the geometry handling
of sysrap, CSG and CSGOptiX packages. Relevant methods/fields include::

stree::get_ridx_type
stree::tri
stree::rem

CSGSolid::setIntent
   invoked by the below importers

CSGImport::importSolid
CSGImport::importSolidGlobal
CSGImport::importSolidFactor

CSGFoundry::isSolidTrimesh
   returns depending on the CSGSolid intent

SBT::createGAS
   depending on CSGFoundry::isSolidTrimesh switches between the mesh and analytic buildInput
   passed to SOPTIX_Accel::Create


In order to dump the triangulation status, best to dump the solids
**/


inline bool stree::is_force_triangulate( int lvid ) const
{
    return slist::Contains( force_triangulate_lvid, lvid );
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

Q: What about ridx:0 is there a factor ?
A: from the freq cut in stree::enumerateFactors and
   the offset by one in stree::labelFactorSubtrees it looks
   like the factor are really just for the instanced

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



/**
stree::findForceTriangulateLVID
--------------------------------

Canonically invoked during stree creation by U4Tree::Create/.../stree::factorize.
Populates *force_triangulate_lvid* vector of indices, which is used by stree::is_force_triangulate.

HMM: the envvar is an input to creation... how to persist that info ?
To make is_force_triangulated work for a loaded stree ?
Just need to persist the vector ?

Uses the optional comma delimited stree__force_triangulate_solid envvar list of unique solid names
together with the member variable vector of all solid names *soname* to form the indices.

When there are many solids that need to be triangulated it is more
convenient to adopt a config approach like the below example using
a replacement path to load the solid names from a file::

    cd ~/.opticks/GEOM

    cp J_2024aug27/CSGFoundry/meshname.txt J_2024aug27_meshname_stree__force_triangulate_solid.txt

        ## copy meshname.txt with all the solid names to a suitable location
        ## "J_2024aug27" is an example GEOM identifier

    vi J_2024aug27_meshname_stree__force_triangulate_solid.txt

        ## edit the file leaving only global solids to be force triangulated
        ## note that as sstr::SplitTrimSuppress is used lines starting with '#' are skipped

    export GEOM=J_2024aug27
    export stree__force_triangulate_solid='filepath:$HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt'

        ## configure envvar within runner script to load the filepath instead of directly using a delimited string
        ## NB the names and paths are just examples

**/


inline void stree::findForceTriangulateLVID()
{
    FindForceTriangulateLVID(force_triangulate_lvid, soname, force_triangulate_solid, ',' );

    if(ssys::getenvbool(_findForceTriangulateLVID_DUMP)) std::cout
        << "stree::findForceTriangulateLVID\n"
        << " [" << _findForceTriangulateLVID_DUMP << "] "
        << descForceTriangulateLVID()
        ;
}

/**
stree::collectGlobalNodes
---------------------------

This is invoked from the tail of stree::factorize
where subsets of the *nds* snode collected from Geant4 by *U4Tree::initNodes_r*
are copied into the *rem* and *tri* vectors.

Done by iterating over the *nds* vector of all nodes copying global non-instanced
snode with repeat_index:0 into the *rem* and *tri* vectors
depending on the "stree__force_triangulate_solid" envvar list of unique solid names.

The default is to collect globals into the *rem* vector.

NB simplifying assumption that all configured tri nodes are global (not instanced)

NB as this action is pre-cache it means that currently must configure triangulation envvar
when doing the from Geant4 to stree.h conversion with U4Tree.h, hence the triangulation
setting configured is baked into the cache geometry

could triangulation action be done post-cache ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* probably yes, but not easily : would need to defer the tri rem formation

should triangulation action be done post-cache ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* probably no, flexibility would be convenient during testing
  but for actual use the definiteness of getting same geometry from cache has value


**/


inline void stree::collectGlobalNodes()
{
    assert( rem.size() == 0u );
    assert( tri.size() == 0u );

    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        const snode& nd = nds[nidx] ;
        assert( nd.index == nidx );
        bool do_force_triangulate = is_force_triangulate(nd.lvid) ;
        if( nd.repeat_index == 0 )
        {
            std::vector<snode>& dst = do_force_triangulate ? tri : rem  ;
            dst.push_back(nd) ;
        }
        else
        {
            assert( do_force_triangulate == false && "force triangulate solid is currently only supported for remainder nodes" );
        }
    }
    if(level>0) std::cout
       << "stree::collectGlobalNodes "
       << descNodes()
       << descForceTriangulateLVID()
       << std::endl
       ;
}


inline std::string stree::descNodes() const
{
    std::stringstream ss ;
    ss
       << "stree::descNodes "
       << " nds.size " << nds.size()
       << " rem.size " << rem.size()
       << " tri.size " << tri.size()
       << " (rem.size + tri.size) " << (rem.size() + tri.size() )
       << " (nds.size - rem.size - tri.size; inst nodes) " << (nds.size() - rem.size() - tri.size() )
       ;
    std::string str = ss.str();
    return str ;
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

findForceTriangulateLVID
   populates force_triangulate_lvid vector of lvid int based on force_triangulate
   envvar and solid names. The vector is used by stree::is_force_triangulate

collectGlobalNodes
   collect global non-instanced nodes into *rem* vector and depending on envvars collect
   nodes to be force triangulated into *tri* vector


**/

inline void stree::factorize()
{
    if(level>0) std::cout << "[ stree::factorize (" << level << ")" << std::endl ;

    classifySubtrees();
    disqualifyContainedRepeats();
    sortSubtrees();
    enumerateFactors();
    labelFactorSubtrees();

    findForceTriangulateLVID();
    collectGlobalNodes();

    if(level>0) std::cout << desc_factor() << std::endl ;
    if(level>0) std::cout << desc_lvid() << std::endl ;

    if(ssys::getenvbool(stree__populate_prim_nidx))
    {
        populate_prim_nidx();
    }
    else if(ssys::getenvbool(stree__populate_nidx_prim))
    {
        populate_prim_nidx();
        populate_nidx_prim();
    }

    if(level>0) std::cout << "] stree::factorize (" << level << ")" << std::endl ;

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
inline int stree::get_triangulate_subtree() const
{
    return tri.size() ;
}

inline int stree::get_remainder_olvid() const
{
    if(rem.size() == 0 ) return -1 ;
    const snode& out = rem[0] ;
    return out.lvid ;
}
inline int stree::get_triangulate_olvid() const
{
    if(tri.size() == 0 ) return -1 ;
    const snode& out = tri[0] ;
    return out.lvid ;
}


inline int stree::get_ridx_subtree(unsigned ridx) const
{
    char ridx_type = get_ridx_type(ridx) ;
    int num_rem = get_num_remainder();
    assert( num_rem == 1 );

    int _subtree = -1 ;
    switch(ridx_type)
    {
        case 'R': _subtree = get_remainder_subtree()             ; break ;
        case 'F': _subtree = get_factor_subtree(ridx - num_rem)  ; break ;
        case 'T': _subtree = get_triangulate_subtree()           ; break ;
    }
    return _subtree ;
}
inline int stree::get_ridx_olvid(unsigned ridx) const
{
    char ridx_type = get_ridx_type(ridx) ;
    int num_rem = get_num_remainder();
    assert( num_rem == 1 );

    int _olvid = -1 ;
    switch(ridx_type)
    {
        case 'R': _olvid = get_remainder_olvid()             ; break ;
        case 'F': _olvid = get_factor_olvid(ridx - num_rem)  ; break ;
        case 'T': _olvid = get_triangulate_olvid()           ; break ;
    }
    return _olvid ;
}

inline int stree::get_num_ridx() const
{
    return get_num_remainder() + get_num_factor() + get_num_triangulated() ;
}

/**
stree::get_num_remainder
-------------------------

Currently always returns 1, as while OptiX geometry might be made to
work with purely instanced compound solids such geometries
are impossible from Geant4 conversions as the world volume would
never be instanced.

Also lots of other places assume always have this

**/

inline int stree::get_num_remainder() const
{
    //return rem.size() > 0 ? 1 : 0 ;
    return 1 ;
}
inline int stree::get_num_triangulated() const
{
    return tri.size() > 0 ? 1 : 0 ;
}

inline int stree::get_num_ridx_(char ridx_type) const
{
    int num = -1 ;
    switch(ridx_type)
    {
        case 'R': num = get_num_remainder()     ; break ;
        case 'F': num = get_num_factor()        ; break ;
        case 'T': num = get_num_triangulated()  ; break ;
    }
    return num ;
}





/**
stree::get_ridx_type
---------------------

The compound solids are assumed to be ordered in the below manner with
the R solid(s) followed by the instanced F solids and then the T solid::

    RFFFFT


Expectation for compound solids:

+-----+-------------------------------+----------------------------------------------------------------------------+
|     | solid type                    |  note                                                                      |
+=====+===============================+============================================================================+
|  R  |  global non-instanced         |   1 solid formed from the many *rem* nodes                                 |
+-----+-------------------------------+----------------------------------------------------------------------------+
|  F  |  factor/instanced             |  0...~10 solids with a few nodes each                                      |
+-----+-------------------------------+----------------------------------------------------------------------------+
|  T  |  triangulated non-instanced   |  0 or 1 solid formed from any *tri* nodes depending on stree envvar config |
+-----+-------------------------------+----------------------------------------------------------------------------+

Indices of ranges of the 3 types of compound solids:

+---+--------------------------------------------+--------------------------------------------------------+
|   |   lowest ridx                              |  highest ridx                                          |
+===+============================================+========================================================+
| R |   0                                        | num_remainder - 1                                      |
+---+--------------------------------------------+--------------------------------------------------------+
| F | num_remainder + 0                          | num_remainder + num_factor -1                          |
+---+--------------------------------------------+--------------------------------------------------------+
| T | num_remainder + num_factor + 0             | num_remainder + num_factor + num_triangulate - 1       |
+---+--------------------------------------------+--------------------------------------------------------+

**/


inline char stree::get_ridx_type(int ridx) const
{
    [[maybe_unused]] int num_ridx = get_num_ridx();
    int num_rem = get_num_remainder();
    int num_fac = get_num_factor();
    int num_tri = get_num_triangulated();

    assert( num_ridx == num_rem + num_fac + num_tri );
    assert( ridx < num_ridx );

    int R[2] = { 0,                     num_rem - 1 } ;
    int F[2] = { num_rem + 0,           num_rem + num_fac - 1 } ;
    int T[2] = { num_rem + num_fac + 0, num_rem + num_fac + num_tri - 1 };

    char type = '?' ;
    if(      ridx >= R[0] && ridx <= R[1] ) type = 'R' ;
    else if( ridx >= F[0] && ridx <= F[1] ) type = 'F' ;
    else if( ridx >= T[0] && ridx <= T[1] ) type = 'T' ;
    return type ;
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
the idx factor.

**/

inline void stree::get_factor_nodes(std::vector<int>& nodes, unsigned idx) const
{
    assert( idx < factor.size() );
    const sfactor& fac = factor[idx];
    std::string sub = fac.get_sub();

    get_nodes(nodes, sub.c_str() );

    bool consistent = int(nodes.size()) == fac.freq ;
    if(!consistent) std::cerr
        << "stree::get_factor_nodes INCONSISTENCY"
        << " nodes.size " << nodes.size()
        << " fac.freq " << fac.freq
        << std::endl
        ;
    assert(consistent );
}

inline std::string stree::desc_factor_nodes(unsigned idx) const
{
    std::vector<int> nodes ;
    get_factor_nodes(nodes, idx);

    int num_nodes = nodes.size();
    std::stringstream ss ;
    ss << "stree::desc_factor_nodes idx " << idx << " num_nodes " << num_nodes << std::endl  ;
    std::string str = ss.str();
    return str ;
}



inline std::string stree::desc_factor() const
{
    std::stringstream ss ;
    ss << "stree::desc_factor" << std::endl << sfactor::Desc(factor) ;
    std::string str = ss.str();
    return str ;
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


/**
stree::get_repeat_field
-------------------------

Iterates over all structural *snode* (aka physical volumes "PV"
collecting field values for selected *snode* where the
selection uses (q_repeat_index, q_repeat_ordinal).)

Supported fields:

  +---------+--------------+-------------------+
  | q_field |  result      |                   |
  +=========+==============+===================+
  |  'I'    |  nd.index    |  get_repeat_nidx  |
  +---------+--------------+-------------------+
  |  'L'    |  nd.lvid     |  get_repeat_lvid  |
  +---------+--------------+-------------------+

**/

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


/**
stree::get_repeat_node
-----------------------

Collect all snode (structual/volumes) selected by (q_repeat_index, q_repeat_ordinal)

Observed this to give zero nodes for the first and last ridx, ie it does
not handle the rem and tri nodes.

::

   TEST=get_repeat_node ~/o/sysrap/tests/stree_load_test.sh

   TEST=get_repeat_node RIDX=1 RORD=10 ~/o/sysrap/tests/stree_load_test.sh run

   TEST=get_repeat_node RIDX=9 RORD=0 ~/o/sysrap/tests/stree_load_test.sh run

**/

inline void stree::get_repeat_node(std::vector<snode>& nodes, int q_repeat_index, int q_repeat_ordinal ) const
{
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        const snode& nd = nds[nidx] ;
        assert( nd.index == nidx );
        if(SelectNode(nd, q_repeat_index, q_repeat_ordinal)) nodes.push_back(nd);
    }
}

/**
stree::desc_repeat_node
------------------------

Dump structural volumes (snode) of the ordinal-th occurrence of q_repeat_index (aka ridx),
collecting unique lvid from all nodes. Then for each unique lvid dump the csg nodes (sn).
This enables for example the CSG shapes within a particular compound solid to be checked.::

    TEST=get_repeat_node RIDX=9 ~/o/sysrap/tests/stree_load_test.sh

**/

inline std::string stree::desc_repeat_node(int q_repeat_index, int q_repeat_ordinal) const
{
    std::vector<snode> nodes ;
    get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal);
    int num_node = nodes.size() ;

    std::stringstream ss ;
    ss << "stree::desc_repeat_nodes"
       << " q_repeat_index " << q_repeat_index
       << " q_repeat_ordinal " << q_repeat_ordinal
       << " num_node " << num_node
       << std::endl
       ;


    std::set<int> ulvid ;

    for(int i=0 ; i < num_node ; i++ )
    {
        const snode& n = nodes[i] ;
        const std::string& lvn = get_lvid_soname_( n.lvid ) ;
        ss << n.desc() << " " << lvn << "\n" ;
        ulvid.insert(n.lvid) ;
    }

    typedef std::set<int>::const_iterator IT ;
    ss << "ulvid {" ;
    for(IT it=ulvid.begin() ; it != ulvid.end() ; it++) ss << *it << "," ;
    ss << "}\n" ;

    for(IT it=ulvid.begin() ; it != ulvid.end() ; it++)
    {
        int lvid = *it ;
        ss << desc_solid(lvid) << "\n" ;
    }

    std::string str = ss.str();
    return str ;
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

    std::string str = ss.str();
    return str ;
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

inline void stree::add_inst_identity( int gas_idx, int nidx )
{
    glm::tmat4x4<double> tr_m2w(1.) ;
    glm::tmat4x4<double> tr_w2m(1.) ;

    add_inst(tr_m2w, tr_w2m, gas_idx, nidx );
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


Note that the inst gas are gauranteed to be
in contiguous tranches, so can just have instcount
and offsets to reference to relevant transforms.

::

    g = i[:,1,3].view(np.int64)

    In [27]: np.where( g == 0 )[0]
    Out[27]: array([0])

    In [28]: np.where( g == 1 )[0]
    Out[28]: array([    1,     2,     3, ..., 25598, 25599, 25600])

    In [29]: np.where( g == 2 )[0]
    Out[29]: array([25601, 25602, 25603, ..., 38213, 38214, 38215])

    In [30]: np.where( g == 3 )[0]
    Out[30]: array([38216, 38217, 38218, ..., 43210, 43211, 43212])

    In [31]: np.where( g == 4 )[0]
    Out[31]: array([43213, 43214, 43215, ..., 45610, 45611, 45612])


**/

inline void stree::add_inst()
{
    int ridx = 0 ;
    int nidx = 0 ;
    int num_inst = 1 ;
    int tot_inst = 0 ;

    add_inst_identity(ridx, nidx );   // global instance with identity transforms

    inst_info.push_back( {ridx,num_inst,tot_inst,0} );
    tot_inst += num_inst  ;


    glm::tmat4x4<double> tr_m2w(1.) ;
    glm::tmat4x4<double> tr_w2m(1.) ;

    unsigned num_factor = get_num_factor();
    for(int i=0 ; i < int(num_factor) ; i++)
    {
        std::vector<int> nodes ;
        get_factor_nodes(nodes, i);

        num_inst = nodes.size();
        ridx = i + 1 ;       // 0 is the global instance, so need this + 1

        if(level > 1) std::cout
            << "stree::add_inst.num_factor "
            << " i " << std::setw(3) << i
            << " ridx(gas_idx) " << std::setw(3) << ridx
            << " num_inst " << std::setw(7) << num_inst
            << std::endl
            ;

        inst_info.push_back( {ridx,num_inst,tot_inst,0} );
        tot_inst += num_inst ;

        for(int j=0 ; j < num_inst ; j++)
        {
            nidx = nodes[j];

            bool local = false ;
            bool reverse = false ;
            get_node_product( tr_m2w, tr_w2m, nidx, local, reverse, nullptr  );

            add_inst(tr_m2w, tr_w2m, ridx, nidx );
        }
    }



    int num_tri = get_num_triangulated();
    assert( num_tri == 0 || num_tri == 1 );

    if(level > 1 ) std::cout
        << "stree::add_inst.num_tri "
        << " num_tri " << std::setw(7) << num_tri
        << std::endl
        ;

    if( num_tri == 1  )
    {
        const snode& tri0 = tri[0] ;

        ridx += 1 ;
        num_inst = 1 ;
        nidx = tri0.index  ; // ? node index of first of the triangulated volumes
        add_inst_identity( ridx, nidx );
        // HMM: is identity transform guaranteed ?

        inst_info.push_back( {ridx,num_inst,tot_inst,0} );
        tot_inst += num_inst ;
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


inline std::string stree::desc_inst_info() const
{
    std::stringstream ss ;
    ss << "[stree::desc_inst_info {ridx, inst_count, inst_offset, 0} " << std::endl ;
    int num_inst_info = inst_info.size();

    int tot_inst = 0 ;
    for(int i=0 ; i < num_inst_info ; i++)
    {
        const int4& info = inst_info[i] ;
        ss
           << "{"
           << std::setw(3) << info.x
           << ","
           << std::setw(7) << info.y
           << ","
           << std::setw(7) << info.z
           << ","
           << std::setw(3) << info.w
           << "}"
           << std::endl
           ;
        tot_inst += info.y ;
    }
    ss << "]stree::desc_inst_info tot_inst " <<  tot_inst << std::endl ;
    std::string str = ss.str();
    return str ;
}

inline std::string stree::desc_inst_info_check() const
{
    int num_gas  = inst_info.size();
    [[maybe_unused]] int num_inst = inst.size();
    int tot = 0 ;
    int tot_count = 0 ;
    for(int i=0 ; i < num_gas ; i++)
    {
        const int4&  _inst_info = inst_info[i] ;
        [[maybe_unused]] int ridx = _inst_info.x ;
        int count = _inst_info.y ;
        int offset = _inst_info.z ;

        tot_count += count ;

        assert( ridx == i );
        for(int j=0 ; j < count ; j++)
        {
            [[maybe_unused]] int idx = offset + j ;
            assert( idx < num_inst );
            assert( idx == tot );
            tot += 1 ;
        }
    }

    std::stringstream ss ;
    ss << "stree::desc_inst_info_check"
       << " tot_count " <<  tot_count
       << std::endl
       << " tot " << tot
       ;
    std::string str = ss.str();
    return str ;
}


/**
stree::find_inst_gas
----------------------

Uses inst_info to provide the instance index of the ordinal-th
instance for the gas_idx

**/

inline int stree::find_inst_gas( int q_gas_idx, int q_gas_ordinal ) const
{
    int num_gas  = inst_info.size();
    bool valid = q_gas_idx < num_gas ;
    if(!valid) return -2 ;

    const int4& _inst_info = inst_info[q_gas_idx] ;

    [[maybe_unused]] int ridx = _inst_info.x ;
    int count = _inst_info.y ;
    int offset = _inst_info.z ;

    assert( ridx == q_gas_idx );

    if( q_gas_ordinal < 0 ) q_gas_ordinal += count ;
    int inst_idx = q_gas_ordinal < count ? offset + q_gas_ordinal : -1 ;
    int num_inst = inst.size();
    return inst_idx < num_inst ? inst_idx : -3  ;
}

inline int stree::find_inst_gas_slowly( int q_gas_idx, int q_gas_ordinal ) const
{
    std::vector<int> v_inst_idx ;
    find_inst_gas_slowly_( v_inst_idx, q_gas_idx );
    int num = v_inst_idx.size() ;
    if( q_gas_ordinal < 0 ) q_gas_ordinal += num ;
    int inst_idx = q_gas_ordinal > -1 && q_gas_ordinal < num ? v_inst_idx[q_gas_ordinal] : -1 ;

    bool dump = false ;
    if(dump && q_gas_idx == 0 ) std::cout
        << "stree::find_inst_gas_slowly"
        << " q_gas_idx " << q_gas_idx
        << " q_gas_ordinal " << q_gas_ordinal
        << " v_inst_idx.size " << v_inst_idx.size()
        << " inst_idx " << inst_idx
        << std::endl
        ;

    return inst_idx ;
}

inline void stree::find_inst_gas_slowly_( std::vector<int>& v_inst_idx , int q_gas_idx ) const
{
    int num_inst = inst.size();
    glm::tvec4<int64_t> col3 ;
    for(int i=0 ; i < num_inst ; i++)
    {
        const glm::tmat4x4<double>& tr_m2w = inst[i] ;
        strid::Decode(tr_m2w, col3 );
        int inst_idx = col3.x ;
        int gas_idx = col3.y ;
        assert( inst_idx == i );
        if( gas_idx == q_gas_idx ) v_inst_idx.push_back(inst_idx) ;
    }
}



inline const glm::tmat4x4<double>* stree::get_inst(int idx) const
{
    return idx > -1 && idx < int(inst.size()) ? &inst[idx] : nullptr ;
}
inline const glm::tmat4x4<double>* stree::get_iinst(int idx) const
{
    return idx > -1 && idx < int(iinst.size()) ? &iinst[idx] : nullptr ;
}

inline const glm::tmat4x4<float>* stree::get_inst_f4(int idx) const
{
    return idx > -1 && idx < int(inst_f4.size()) ? &inst_f4[idx] : nullptr ;
}
inline const glm::tmat4x4<float>* stree::get_iinst_f4(int idx) const
{
    return idx > -1 && idx < int(iinst_f4.size()) ? &iinst_f4[idx] : nullptr ;
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

/**
stree::add_extra_surface
-------------------------

If the name is already present in the suname list
just returns the 0-based index otherwise add to suname
and return the new index.

**/

inline int stree::add_extra_surface( const char* name )
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
        //suindex.push_back(idx);
        int idx2 = stree::GetValueIndex<std::string>( suname, name ) ;
        bool idx_expect = idx2 == idx ;
        assert( idx_expect );
        if(!idx_expect) std::raise(SIGINT);
    }
    return idx ;
}


inline int stree::add_extra_surface(const std::vector<std::string>& names  )
{
    int idx = -1 ;
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* surname = names[i].c_str() ;
        idx = add_extra_surface( surname );
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




/**
stree::add_surface_implicit
----------------------------

Used from U4TreeBorder::get_implicit_idx

THIS FORMERLY RETURNED THE IMPLICIT_IDX,
NOW RETURNING THE STANDARD SURFACE IDX

**/


inline int stree::add_surface_implicit( const char* name )
{
    int idx = add_extra_surface(name);

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
        bool boundary_expect = num_bd_check == boundary ;
        assert( boundary_expect );
        if(!boundary_expect) std::raise(SIGINT);
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

    init_material_mapping() ;
}


/**
stree::init_material_mapping
-------------------------------

Formerly similar to this was done by stree::import_bnd
but need the material map for live geometry running,
so repositioned to being invoked from stree::initStandard

**/


inline void stree::init_material_mapping()
{
    assert( mtline.size() == 0 );
    assert( mtname.size() == mtindex.size() );

    // for each mtname use bnd->names to fill the mtline vector
    SBnd::FillMaterialLine( mtline, mtindex, mtname, bdname );

    // fill (int,int) map from the mtline and mtindex vectors
    init_mtindex_to_mtline() ;

    if(ssys::getenvbool(_init_material_mapping_DUMP)) std::cerr
        << "stree::init_material_mapping"
        << " [" << _init_material_mapping_DUMP <<  "] "
        << " desc_mt "
        << std::endl
        << desc_mt()
        << std::endl
        ;
}





/**
stree::import_bnd
-------------------

Moved from SSim::import_bnd

The mtname and mtindex are populated by stree::add_material,
beyond those just need the bnd names to determine the
mtline and the map.


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

**/




/**
stree::init_mtindex_to_mtline
------------------------------

SUSPECT THIS IS ONLY CALLED ON LOADING : NOT
ON CREATION : CAUSING CERENKOV MTLINE ISSUE

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


/**
stree::populate_prim_nidx
----------------------------

mapping stree.h/nidx to CSGFoundry/globalPrimIdx ?

Need to "faux_import" ie a stripped down form of CSGImport::importSolid
within stree to establish the correspondence between nidx and globalPrimIdx
Every nidx will have a single globalPrimIdx but its not 1:1
multiple nidx will have the same globalPrimIdx due to instancing.

HMM this approach provides way to go from globalPrimIdx to nidx,
but not the general case of the reverse for instanced nodes.
That might not matter as the main usefulness of globalPrimIdx
is for the global geometry. Instanced geometry is best
identified with the instance index.

**/

inline void stree::populate_prim_nidx()
{
    faux_importSolid();
}

inline void stree::faux_importSolid()
{
    // follow the pattern of CSGImport::importSolid
    int num_ridx = get_num_ridx() ;
    for(int ridx=0 ; ridx < num_ridx ; ridx++)
    {
        char ridx_type = get_ridx_type(ridx) ;
        switch(ridx_type)
        {
            case 'R': faux_importSolidGlobal( ridx, ridx_type ) ; break ;   // remainder
            case 'T': faux_importSolidGlobal( ridx, ridx_type ) ; break ;   // triangulate
            case 'F': faux_importSolidFactor( ridx, ridx_type ) ; break ;   // factor
        }
    }
}
inline void stree::faux_importSolidGlobal(int ridx, char ridx_type )
{
    assert( ridx_type == 'R' || ridx_type == 'T' );  // remainder or triangulate
    const std::vector<snode>* src = get_node_vector(ridx_type) ;
    assert( src );
    int num_src = src->size() ;

    for(int i=0 ; i < num_src ; i++)
    {
        int primIdx = i ;  // primIdx within the CSGSolid
        const snode& node = (*src)[primIdx] ;
        int globalPrimIdx = faux_importPrim( primIdx, node ) ;
        assert( globalPrimIdx >= 0 );
    }
}


inline void stree::faux_importSolidFactor(int ridx, char ridx_type )
{
    assert( ridx > 0 );
    assert( ridx_type == 'F' );

    int  num_rem = get_num_remainder() ;
    assert( num_rem == 1 ) ;  // YEP: always one

    int num_factor = factor.size() ;
    assert( ridx - num_rem < num_factor );

    const sfactor& sf = factor[ridx-num_rem] ;
    int subtree = sf.subtree ;  // number of prim within the compound solid


    int q_repeat_index = ridx ;
    int q_repeat_ordinal = 0 ;   // just first repeat

    std::vector<snode> nodes ;
    get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ;

    std::cout
        << " stree::faux_importSolidFactor "
        << " ridx " << ridx
        << " ridx_type " << ridx_type
        << " num_rem " << num_rem
        << " num_factor " << num_factor
        << " nodes.size " << nodes.size()
        << " subtree " << subtree
        << "\n"
        ;

    assert( subtree == int(nodes.size()) );

    for(int i=0 ; i < subtree ; i++)
    {
        int primIdx = i ;  // primIdx within the CSGSolid
        const snode& node = nodes[primIdx] ;   // structural node
        int globalPrimIdx = faux_importPrim( primIdx, node );
        assert( globalPrimIdx >= 0 );
    }
}

inline int stree::faux_importPrim(int primIdx, const snode& node )
{
    int globalPrimIdx = prim_nidx.size();
    prim_nidx.push_back(node.index);
    return globalPrimIdx ;
}


/**
stree::search_prim_for_nidx_first
---------------------------------

Only expected to give globalPrimIdx for global or first instance nidx,
for the rest of the repeated nidx will give -1

**/

inline int stree::search_prim_for_nidx_first(int nidx) const
{
    size_t gpi = std::distance( prim_nidx.begin(), std::find(prim_nidx.begin(), prim_nidx.end(), nidx ));
    return gpi < prim_nidx.size() ? int(gpi) : -1  ;
}



/**
stree::populate_nidx_prim  WIP : needs shakedown on full geometry
-------------------------------------------------------------------

Needs to be called after populate_prim_nidx

**/


inline void stree::populate_nidx_prim()
{
    nidx_prim.resize(nds.size());
    std::fill( nidx_prim.begin(), nidx_prim.end(), -1 );

    int num_ridx = get_num_ridx() ;
    for(int ridx=0 ; ridx < num_ridx ; ridx++)
    {
        //char ridx_type = get_ridx_type(ridx) ;
        int q_repeat_index = ridx ;

        std::vector<snode> nodes_first ;
        get_repeat_node(nodes_first, q_repeat_index, 0) ;
        int num_nodes_first = nodes_first.size();

        // first repeat nodes should all have corresponding prim
        std::vector<int> gpi_first(num_nodes_first);
        for(int i=0 ; i < num_nodes_first ; i++)
        {
            int gpi = search_prim_for_nidx_first(nodes_first[i].index);
            assert( gpi > -1 );
            gpi_first[i] = gpi ;
        }

        std::vector<snode> nodes_all ;
        get_repeat_node(nodes_all, q_repeat_index, -2) ;
        int num_nodes_all = nodes_all.size();
        // hmm can getting these in regular pattern be relied on
        // if so can use that pattern to pass the prim
        // from the first to all the repeats

        if(num_nodes_first == 0)
        {
            for(int i=0 ; i < num_nodes_all ; i++)
            {
                int nidx = nodes_all[i].index ;
                int gpi = search_prim_for_nidx_first(nidx);
                nidx_prim[nidx] = gpi ;
            }
        }
        else if(num_nodes_first > 0)
        {
            assert( num_nodes_all % num_nodes_first == 0 );
            int num_repeat = num_nodes_all/num_nodes_first ;

            std::cout
                << "stree::populate_nidx_prim"
                << " ridx " << ridx
                << " num_nodes_first " << num_nodes_first
                << " num_nodes_all " << num_nodes_all
                << " num_repeat " << num_repeat
                << "\n"
                ;


            for(int j=0 ; j < num_nodes_all ; j++)
            {
                int i = j % num_nodes_first ;  // maybe ?
                int gpi0 = gpi_first[i] ;

                int nidx = nodes_all[j].index ;
                nidx_prim[nidx] = gpi0 ;
            }

        }


    }
}

/**
stree::get_prim_for_nidx
---------------------------------

Expected to give globalPrimIdx for all nidx,
never giving -1 for valid nidx.

**/

inline int stree::get_prim_for_nidx(int nidx) const
{
    return nidx < int(nidx_prim.size()) ? nidx_prim[nidx] : -1 ;
}


