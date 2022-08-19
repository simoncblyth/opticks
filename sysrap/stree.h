#pragma once
/**
stree.h : explore minimal approach to geometry translation
============================================================

See also u4/U4Tree.h that creates the stree from Geant4 volumes. 

* this is seeking to replace lots of GGeo code, notably: GInstancer.cc GNode.cc 

* DONE : controlling the order of repeats with same freq, using 2-level sort  


TODO:

* ridx labelling the tree
* maintain correspondence between source nodes and destination nodes thru the factorization
* triplet_identity 
* transform rebase

mapping from "factorized" instances back to origin PV and vice-versa 
-----------------------------------------------------------------------

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

Actually the natural place to keep map back info for the remainder 
is withing the CSGPrim.  Normally use of that is for identity purposes is restricted 
because tthe CSGPrim are references from all the instances but for the remainder
the CSGPrim are only referenced once (?). TODO: check this 

TODO: collect the nidx of the remainder into stree.h ?


**/

#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "NP.hh"
#include "NPFold.h"

#include "snode.h"
#include "sdigest.h"
#include "sfreq.h"
#include "sstr.h"
#include "strid.h"
#include "sfactor.h"



struct stree
{
    static constexpr const int MAXDEPTH = 15 ; // presentational only   
    static constexpr const int FREQ_CUT = 500 ;   // HMM GInstancer using 400   
    // subtree digests with less repeats than FREQ_CUT within the entire geometry 
    // are not regarded as repeats for instancing factorization purposes 

    static constexpr const char* RELDIR = "stree" ;
    static constexpr const char* NDS = "nds.npy" ;
    static constexpr const char* M2W = "m2w.npy" ;
    static constexpr const char* W2M = "w2m.npy" ;
    static constexpr const char* GTD = "gtd.npy" ;  // GGeo transform debug, populated in X4PhysicalVolume::convertStructure_r 
    static constexpr const char* MTNAME = "mtname.txt" ;
    static constexpr const char* MTINDEX = "mtindex.npy" ;
    static constexpr const char* MTLINE = "mtline.npy" ;
    static constexpr const char* SONAME = "soname.txt" ;
    static constexpr const char* DIGS = "digs.txt" ;
    static constexpr const char* SUBS = "subs.txt" ;
    static constexpr const char* SUBS_FREQ = "subs_freq" ;
    static constexpr const char* MTFOLD = "mtfold" ;
    static constexpr const char* FACTOR = "factor.npy" ;
    static constexpr const char* INST = "inst.npy" ; 
    static constexpr const char* IINST = "iinst.npy" ; 
    static constexpr const char* INST_F4 = "inst_f4.npy" ; 
    static constexpr const char* IINST_F4 = "iinst_f4.npy" ; 
    static constexpr const char* SENSOR_ID = "sensor_id.npy" ; 
    static constexpr const char* INST_NIDX = "inst_nidx.npy" ; 

    std::vector<std::string> mtname ;       // unique material names
    std::vector<int>         mtindex ;      // G4Material::GetIndex 0-based creation indices 
    std::vector<int>         mtline ;     
    std::map<int,int>        mtindex_to_mtline ;  // not persisted, filled from mtindex and mtline with init_mtindex_to_mtline


    std::vector<std::string> soname ;       // unique solid names

    std::vector<glm::tmat4x4<double>> m2w ; // model2world transforms for all nodes
    std::vector<glm::tmat4x4<double>> w2m ; // world2model transforms for all nodes  
    std::vector<glm::tmat4x4<double>> gtd ; // GGeo Transform Debug, added from X4PhysicalVolume::convertStructure_r


    std::vector<snode> nds ;                // snode info for all nodes
    std::vector<std::string> digs ;         // per-node digest for all nodes  
    std::vector<std::string> subs ;         // subtree digest for all nodes
    std::vector<sfactor> factor ;          // small number of unique subtree factor, digest and freq  
    std::vector<int> sensor_id ;           // updated by reorderSensors

    int level ; 
    unsigned sensor_count ; 
    sfreq* subs_freq ;                      // occurence frequency of subtree digests in entire tree 
    NPFold* mtfold ;                        // material properties

    // HMM: the stree.h inst members and methods are kinda out-of-place
    //      as CSGFoundry already has inst : so the below are looking ahead 
    //      to what will be done by a future "CSGFoundry::CreateFromSTree"  

    std::vector<glm::tmat4x4<double>> inst ; 
    std::vector<glm::tmat4x4<float>>  inst_f4 ; 
    std::vector<glm::tmat4x4<double>> iinst ; 
    std::vector<glm::tmat4x4<float>>  iinst_f4 ; 

    std::vector<int>                  inst_nidx ; 


    stree();

    std::string desc() const ;
    std::string desc_vec() const ;
    std::string desc_sub(bool all=false) const ;
    std::string desc_sub(const char* sub) const ;

    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr );
    static std::string Digest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ;   // immediate children
    void get_progeny( std::vector<int>& progeny, int nidx ) const ;   // recursively get children and all their children and so on... 
    std::string desc_progeny(int nidx) const ; 

    void traverse(int nidx=0) const ; 
    void traverse_r(int nidx, int depth, int sibdex) const ; 

    void reorderSensors(); 
    void reorderSensors_r(int nidx); 
    void get_sensor_id( std::vector<int>& sensor_id ) const ; 
    std::string desc_sensor_id(unsigned edge=10) const ; 
    static std::string DescSensor( const std::vector<int>& sensor_id, const std::vector<int>& sensor_idx, unsigned edge=10 ); 

    void lookup_sensor_identifier( 
         std::vector<int>& arg_sensor_identifier, 
         const std::vector<int>& arg_sensor_index, 
         bool one_based_index, bool verbose=false, unsigned edge=10 ) const ; 

    sfreq* make_progeny_freq(int nidx) const ; 
    sfreq* make_freq(const std::vector<int>& nodes ) const ; 

    int  find_lvid(const char* soname_, bool starting=true  ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, int lvid ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, const char* soname_, bool starting=true ) const ; 
    int  find_lvid_node( const char* q_soname, int ordinal ) const ; 
    int  find_lvid_node( const char* q_spec ) const ; // eg HamamatsuR12860sMask_virtual:0:1000

    void get_sub_sonames( std::vector<std::string>& sonames ) const ; 
    const char* get_sub_soname(const char* sub) const ; 

    const char* get_soname(int nidx) const ; 
    const char* get_sub(   int nidx) const ; 
    int         get_depth( int nidx) const ; 
    int         get_parent(int nidx) const ; 
    int         get_lvid(  int nidx) const ; 
    int         get_copyno(int nidx) const ; 

    const glm::tmat4x4<double>& get_m2w(int nidx) const ; 
    const glm::tmat4x4<double>& get_w2m(int nidx) const ; 

    void get_ancestors(  std::vector<int>& ancestors, int nidx ) const ;
    void get_m2w_product( glm::tmat4x4<double>& transform, int nidx, bool reverse ) const ; // expect reverse:false 
    std::string desc_m2w_product(int nidx, bool reverse) const ; 

    void get_w2m_product( glm::tmat4x4<double>& transform, int nidx, bool reverse ) const ; // expect reverse:true 

    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ; 


    std::string subtree_digest( int nidx ) const ;
    static std::string depth_spacer(int depth); 

    std::string desc_node_(int nidx, const sfreq* sf ) const ;
    std::string desc_node(int nidx, bool show_sub=false) const ; 

    std::string desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems=10) const ;
    std::string desc_ancestry(int nidx, bool show_sub=false) const ;

    static std::string FormPath(const char* base, const char* rel ); 

    void save_( const char* fold ) const ;
    void save( const char* base, const char* reldir=RELDIR ) const ;

    template<typename S, typename T>   // S:compound type T:atomic "transport" type
    static void ImportArray( std::vector<S>& vec, const NP* a ); 

    void load_( const char* fold );
    void load( const char* base, const char* reldir=RELDIR );
    static stree* Load(const char* base, const char* reldir=RELDIR ); 

    static int Compare( const std::vector<int>& a, const std::vector<int>& b ) ; 
    static std::string Desc(const std::vector<int>& a, unsigned edgeitems=10 ) ; 


    void classifySubtrees();
    bool is_contained_repeat(const char* sub) const ; 
    void disqualifyContainedRepeats();
    void sortSubtrees(); 
    void enumerateFactors(); 
    void labelFactorSubtrees(); 

    void factorize(); 

    unsigned get_num_factor() const ; 
    sfactor& get_factor(unsigned idx); 
    void get_factor_nodes(std::vector<int>& nodes, unsigned idx) const ; 
    std::string desc_factor() const ; 

    void add_inst( glm::tmat4x4<double>& m2w, glm::tmat4x4<double>& w2m, int gas_idx, int nidx ); 
    void add_inst(); 
    void narrow_inst(); 
    void clear_inst(); 
    std::string desc_inst() const ;

    void get_mtindex_range(int& mn, int& mx ) const ; 
    std::string desc_mt() const ; 

    void add_material( const char* name, unsigned g4index ); 
    void init_mtindex_to_mtline(); 
    int lookup_mtline( int mtindex ) const ; 

};



inline stree::stree()
    :
    level(1),        // set to 0: once operational
    sensor_count(0),
    subs_freq(new sfreq),
    mtfold(new NPFold)
{
}

inline std::string stree::desc() const
{
    std::stringstream ss ;
    ss << "stree::desc"
       << " sensor_count " << sensor_count 
       << " nds " << nds.size()
       << " m2w " << m2w.size()
       << " w2m " << w2m.size()
       << " gtd " << gtd.size()
       << " digs " << digs.size()
       << " subs " << subs.size()
       << " soname " << soname.size()
       << " subs_freq " << std::endl
       << ( subs_freq ? subs_freq->desc() : "-" )
       << " mtfold " << ( mtfold ? mtfold->desc() : "-" )
       << std::endl
       ;

    std::string s = ss.str();
    return s ;
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
    std::cout 
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

This attempts to mimic the preorder traverse sensor order 
used by GGeo/CSG_GGeo to facilitate comparison. 

When invoked this changes the nd.sensor_index compared 
to the initial ordering of U4Tree::identifySensitiveInstances

Note that this yields a 0-based sensor index. 

**/

inline void stree::reorderSensors()
{
    std::cout << "[ stree::reorderSensors" << std::endl ; 
    sensor_count = 0 ; 
    reorderSensors_r(0); 
    std::cout << "] stree::reorderSensors sensor_count " << sensor_count << std::endl ; 

    get_sensor_id(sensor_id); 
    assert( sensor_count == sensor_id.size() ); 
}
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
stree::get_sensor_id
----------------------

List *sensor_id* obtained by iterating over all *nds* of the geometry.
As the *nds* vector is in preorder traversal order, the order of 
the *sensor_id* should correspond to *sensor_index* from 0 to num_sensor-1. 

**/

inline void stree::get_sensor_id( std::vector<int>& sensor_id ) const 
{
    sensor_id.clear(); 
    for(unsigned nidx=0 ; nidx < nds.size() ; nidx++)
    {
        const snode& nd = nds[nidx] ; 
        if( nd.sensor_id > -1 ) sensor_id.push_back(nd.sensor_id) ; 
    }
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

The arg_sensor_identifier array is resized to match arg_sensor_index and 
populated with sensor_id values. 

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
        int s_index = one_based_index ? arg_sensor_index[i] - 1 : arg_sensor_index[i] ;  // "correct" 1-based to be 0-based 
        bool s_index_inrange = s_index > -1 && s_index < int(sensor_id.size()) ; 
        int s_identifier = s_index_inrange ? sensor_id[s_index] : -1 ; 

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
        }

        arg_sensor_identifier[i] = s_identifier ; 
    }   
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

inline const glm::tmat4x4<double>& stree::get_m2w(int nidx) const 
{
    assert( nidx > -1 && nidx < int(m2w.size())); 
    return m2w[nidx] ; 
}

inline const glm::tmat4x4<double>& stree::get_w2m(int nidx) const 
{
    assert( nidx > -1 && nidx < int(w2m.size())); 
    return w2m[nidx] ; 
}

/**
stree::get_ancestors
---------------------

Collects parent, then parent-of-parent and so on 
until reaching root (nidx:0) which has no parent.  
Then reverses the list to put into root first order. 

This should work even during node collection immediately 
after the parent link has been set and the snode pushed back. 

At first glance you might think this would miss root, but that 
is not the case as it collects parents and the node prior 
to the parent results in collecting root nidx:0. 

**/

inline void stree::get_ancestors( std::vector<int>& ancestors, int nidx ) const
{
    int parent = get_parent(nidx) ;  // simple node lookup nds[nidx].parent 
    while( parent > -1 )
    {
        ancestors.push_back(parent);
        parent = get_parent(parent);
    }
    std::reverse( ancestors.begin(), ancestors.end() );
}

/**
stree::get_m2w_product
------------------------

As this uses get_ancestors (which operates via parent links) and 
get_m2w it should work during node creation immediately after 
the snode and m2w transforms are pushed back. 


Note that even when things appear to be working OK, 
bugs can still lurk as lots of the transforms in the stack are identity. 

So there is still potential for wrong direction of multiplication bugs. 

**/

inline void stree::get_m2w_product( glm::tmat4x4<double>& transform, int nidx, bool reverse ) const 
{
    std::vector<int> nodes ; 
    get_ancestors(nodes, nidx); 
    nodes.push_back(nidx); 

    unsigned num_nodes = nodes.size();
    glm::tmat4x4<double> xform(1.); 

    for(unsigned i=0 ; i < num_nodes ; i++ ) 
    {
        int idx = nodes[reverse ? num_nodes - 1 - i : i] ; 
        const glm::tmat4x4<double>& t = get_m2w(idx) ; 
        xform *= t ;           
    }
    assert( sizeof(glm::tmat4x4<double>) == sizeof(double)*16 ); 
    memcpy( glm::value_ptr(transform), glm::value_ptr(xform), sizeof(glm::tmat4x4<double>) );
}


inline std::string stree::desc_m2w_product(int nidx, bool reverse) const 
{
    std::vector<int> nodes ; 
    get_ancestors(nodes, nidx); 
    nodes.push_back(nidx); 
    unsigned num_nodes = nodes.size();

    std::stringstream ss ; 
    ss << "stree::desc_m2w_product"
       << " nidx " << nidx 
       << " reverse " << reverse
       << " num_nodes " << num_nodes 
       << " nodes [" 
       ; 
    for(unsigned i=0 ; i < num_nodes ; i++ ) ss << " " << nodes[i] ; 
    ss << "]" << std::endl ; 

    glm::tmat4x4<double> xform(1.); 
    for(unsigned i=0 ; i < num_nodes ; i++ ) 
    {
        int idx = nodes[reverse ? num_nodes - 1 - i : i] ; 
        const glm::tmat4x4<double>& t = get_m2w(idx) ; 
        xform *= t ; 

        const char* so = get_soname(idx) ; 
        ss << " i " << i << " idx " << idx << " so " << so << std::endl ; 
        ss << strid::Desc_("t", "xform",  t, xform ) << std::endl ; 
    }

    std::string s = ss.str(); 
    return s ; 
}


/**
stree::get_w2m_product
-----------------------

**/

inline void stree::get_w2m_product( glm::tmat4x4<double>& transform, int nidx, bool reverse ) const 
{
    std::vector<int> nodes ; 
    get_ancestors(nodes, nidx); 
    nodes.push_back(nidx); 
    unsigned num_nodes = nodes.size();
    glm::tmat4x4<double> xform(1.); 

    for(unsigned i=0 ; i < nodes.size() ; i++ ) 
    {
        int idx = nodes[reverse ? num_nodes - 1 - i : i] ; 
        const glm::tmat4x4<double>& t = get_w2m(idx) ;  
        xform *= t ; 
    }
    assert( sizeof(glm::tmat4x4<double>) == sizeof(double)*16 ); 
    memcpy( glm::value_ptr(transform), glm::value_ptr(xform), sizeof(glm::tmat4x4<double>) );
}

/**
stree::get_nodes
------------------

Collects node indices of all nodes with the subtree digest provided in the argument. 
So the "outer volume" nodes are returned. 

**/

inline void stree::get_nodes(std::vector<int>& nodes, const char* sub) const
{
    for(unsigned i=0 ; i < subs.size() ; i++) if(strcmp(subs[i].c_str(), sub)==0) nodes.push_back(int(i)) ;
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

    std::stringstream ss ;
    ss << depth_spacer(nd.depth) ; 
    ss << nd.desc() ;
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

inline std::string stree::desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems) const
{
    std::stringstream ss ;
    ss << "stree::desc_nodes " << vnidx.size() << std::endl ;
    for(unsigned i=0 ; i < vnidx.size() ; i++)
    {
        if( i < edgeitems || ( i > vnidx.size() - edgeitems ))
            ss << desc_node(vnidx[i]) << std::endl ;
        else if( i == edgeitems ) ss << " ... " << std::endl ;
    }
    std::string s = ss.str();
    return s ;
}

inline std::string stree::desc_ancestry(int nidx, bool show_sub) const
{
    std::vector<int> ancestors ;
    get_ancestors(ancestors, nidx);

    std::stringstream ss ; 
    ss << "stree::desc_ancestry nidx " << nidx << std::endl ;

    for(unsigned i=0 ; i < ancestors.size() ; i++)
    {
        int ix = ancestors[i] ;
        ss << desc_node(ix, show_sub) << std::endl ;
    }
    ss << std::endl ;
    ss << desc_node(nidx, show_sub) << " " << std::endl ; 

    std::string s = ss.str();
    return s ; 
}






inline std::string stree::FormPath(const char* base, const char* rel ) // static
{
    std::stringstream ss ;    
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    return dir ; 
}

inline void stree::save( const char* base, const char* rel ) const 
{
    std::string dir = FormPath(base, rel); 
    save_(dir.c_str()); 
}

inline void stree::save_( const char* fold ) const 
{
    std::cout << "stree::save_ " << ( fold ? fold : "-" ) << std::endl ; 
    NP::Write<int>(    fold, NDS, (int*)nds.data(),    nds.size(), snode::NV );
    NP::Write<double>( fold, M2W, (double*)m2w.data(), m2w.size(), 4, 4 );
    NP::Write<double>( fold, W2M, (double*)w2m.data(), w2m.size(), 4, 4 );
    NP::Write<double>( fold, GTD, (double*)gtd.data(), gtd.size(), 4, 4 );
    NP::WriteNames(    fold, MTNAME,   mtname );
    NP::Write<int>(    fold, MTINDEX, (int*)mtindex.data(),  mtindex.size() );
    NP::Write<int>(    fold, MTLINE,  (int*)mtline.data(),   mtline.size() );
    NP::WriteNames( fold, SONAME, soname );
    NP::WriteNames( fold, DIGS,   digs );
    NP::WriteNames( fold, SUBS,   subs );
    if(subs_freq) subs_freq->save(fold, SUBS_FREQ);
    NP::Write<int>(fold, FACTOR, (int*)factor.data(), factor.size(), sfactor::NV ); 
    if(mtfold) mtfold->save(fold, MTFOLD) ; 


    NP::Write<double>(fold,  INST,     (double*)inst.data(), inst.size(), 4, 4 ); 
    NP::Write<double>(fold,  IINST,    (double*)iinst.data(), iinst.size(), 4, 4 ); 
    // _f4 just for debug comparisons : narrowing normally only done in memory for upload  
    NP::Write<float>(fold,  INST_F4,  (float*)inst_f4.data(), inst_f4.size(), 4, 4 ); 
    NP::Write<float>(fold,  IINST_F4, (float*)iinst_f4.data(), iinst_f4.size(), 4, 4 ); 

    assert( sensor_count == sensor_id.size() ); 
    NP::Write<int>(    fold, SENSOR_ID, (int*)sensor_id.data(), sensor_id.size() );
    NP::Write<int>(    fold, INST_NIDX, (int*)inst_nidx.data(), inst_nidx.size() );

}

inline void stree::load( const char* base, const char* reldir ) 
{
    std::string dir = FormPath(base, reldir ); 
    load_(dir.c_str()); 
}

inline stree* stree::Load(const char* base, const char* reldir ) // static 
{
    stree* st = new stree ; 
    st->load(base, reldir); 
    return st ; 
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
template void stree::ImportArray<glm::tmat4x4<double>, double>(std::vector<glm::tmat4x4<double>>& , const NP* ); 
template void stree::ImportArray<glm::tmat4x4<float>, float>(std::vector<glm::tmat4x4<float>>& , const NP* ); 
template void stree::ImportArray<sfactor, int>(std::vector<sfactor>& , const NP* ); 


inline void stree::load_( const char* fold )
{
    std::cout << "stree::load_ " << ( fold ? fold : "-" ) << std::endl ; 

    ImportArray<snode, int>(                  nds, NP::Load(fold, NDS)); 
    ImportArray<glm::tmat4x4<double>, double>(m2w, NP::Load(fold, M2W)); 
    ImportArray<glm::tmat4x4<double>, double>(w2m, NP::Load(fold, W2M)); 
    ImportArray<glm::tmat4x4<double>, double>(gtd, NP::Load(fold, GTD)); 

    NP::ReadNames( fold, SONAME, soname );
    NP::ReadNames( fold, MTNAME, soname );

    ImportArray<int, int>( mtindex, NP::Load(fold, MTINDEX) );
    ImportArray<int, int>( mtline,  NP::Load(fold, MTLINE) );
    init_mtindex_to_mtline(); 

    NP::ReadNames( fold, DIGS,   digs );
    NP::ReadNames( fold, SUBS,   subs );

    if(subs_freq) subs_freq->load(fold, SUBS_FREQ) ;
    ImportArray<sfactor, int>( factor, NP::Load(fold, FACTOR) ); 
    if(mtfold) mtfold->load(fold, MTFOLD) ;

    ImportArray<glm::tmat4x4<double>, double>(inst,   NP::Load(fold, INST)); 
    ImportArray<glm::tmat4x4<double>, double>(iinst,  NP::Load(fold, IINST)); 
    ImportArray<glm::tmat4x4<float>, float>(inst_f4,  NP::Load(fold, INST_F4)); 
    ImportArray<glm::tmat4x4<float>, float>(iinst_f4, NP::Load(fold, IINST_F4)); 

    ImportArray<int, int>( sensor_id, NP::Load(fold, SENSOR_ID) );
    sensor_count = sensor_id.size(); 

    ImportArray<int, int>( inst_nidx, NP::Load(fold, INST_NIDX) );
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

Its necessary to use two level ordering 
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
    if(level > 0) std::cout << "[ stree::enumerateFactors " << std::endl ;
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

        int fac_subtree = -1 ; 
        for(unsigned i=0 ; i < outer_node.size() ; i++)
        {
            int outer = outer_node[i] ; 
            std::vector<int> subtree ; 
            get_progeny(subtree, outer); 
            subtree.push_back(outer); 

            if(fac_subtree == -1) 
            {
                fac_subtree = subtree.size() ;  
            }
            else 
            {
                assert( int(subtree.size()) == fac_subtree ); 
            }

            for(unsigned i=0 ; i < subtree.size() ; i++)
            {
                int nidx = subtree[i] ; 
                snode& nd = nds[nidx] ; 
                assert( nd.index == nidx ); 
                nd.repeat_index = repeat_index ; 
            }
        }
        fac.subtree = fac_subtree ; 

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

**/

inline void stree::factorize()
{
    if(level>0) std::cout << "[ stree::factorize " << std::endl ;
    classifySubtrees(); 
    disqualifyContainedRepeats();
    sortSubtrees(); 
    enumerateFactors(); 
    labelFactorSubtrees(); 
    if(level>0) std::cout << "] stree::factorize " << std::endl ;
}


inline unsigned stree::get_num_factor() const
{
    return factor.size(); 
}

inline sfactor& stree::get_factor(unsigned idx) 
{
    assert( idx < factor.size() ); 
    return factor[idx] ; 
}

/**
stree::get_factor_nodes
--------------------------

Get node indices of the *idx* factor (0-based)

**/

inline void stree::get_factor_nodes(std::vector<int>& nodes, unsigned idx) const 
{
    assert( idx < factor.size() ); 
    const sfactor& fac = factor[idx]; 
    std::string sub = fac.get_sub(); 
    int freq = fac.freq ; 

    get_nodes(nodes, sub.c_str() );  

    bool consistent = int(nodes.size()) == freq ; 
    if(!consistent) 
        std::cerr 
            << "stree::get_factor_nodes INCONSISTENCY"
            << " nodes.size " << nodes.size()
            << " freq " << freq 
            << std::endl 
            ;

    assert(consistent );   
}


inline std::string stree::desc_factor() const 
{
    unsigned num_factor = factor.size(); 
    std::stringstream ss ; 
    ss << "stree::desc_factor"
       << " get_num_factor " 
       << num_factor 
       << std::endl 
       ;

    for(unsigned idx=0 ; idx < num_factor ; idx++) 
    {
        const sfactor& fac = factor[idx]; 
        ss << fac.desc() << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}


/**
stree::add_inst
----------------

Canonically invoked from U4Tree::Create 

**/

inline void stree::add_inst( glm::tmat4x4<double>& tr_m2w,  glm::tmat4x4<double>& tr_w2m, int gas_idx, int nidx )
{
    assert( nidx > -1 && nidx < int(nds.size()) ); 
    const snode& nd = nds[nidx]; 
    int ins_idx = int(inst.size());     // follow sqat4.h::setIdentity

    /*
    TODO: include the nidx into the 4th column ints  OR keep it separately 

    HMM: initial thinking is to squeeze identity info into 32 bits 
    so it can survive being narrowed 

    unsigned nidx_gidx = (( nidx & 0xffffff ) << 8 ) | ( gas_idx & 0xff ) ;  
    union uif64_t {
        uint64_t  u ; 
        int64_t   i ; 
        double    f ; 
    };  
    uif64_t uif ; 
    uif.u = nidx_gidx ; 
    */

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
            get_m2w_product(tr_m2w, nidx, false); 
            get_w2m_product(tr_w2m, nidx, true ); 

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

    std::string s = ss.str(); 
    return s ; 
}



/**
stree::add_material
----------------------

g4index is the Geant4 creation index obtained from G4Material::GetIndex

**/

inline void stree::add_material( const char* name, unsigned g4index )
{
    mtname.push_back(name); 
    mtindex.push_back(g4index); 
} 


/**
stree::init_mtindex_to_mtline
------------------------------

Canonically invoked from SSim::import_bnd/SBnd::FillMaterialLine following 
live creation or from stree::load_ when loading a persisted stree.  

**/

inline void stree::init_mtindex_to_mtline()
{
    bool consistent = mtindex.size() == mtline.size() ;
    if(!consistent) std::cerr << "must use SBnd::fillMaterialLine once have bnd specs" << std::endl ; 
    assert(consistent); 
    for(unsigned i=0 ; i < mtindex.size() ; i++) mtindex_to_mtline[mtindex[i]] = mtline[i] ;  
}

inline int stree::lookup_mtline( int mtindex ) const 
{
    return mtindex_to_mtline.count(mtindex) == 0 ? -1 :  mtindex_to_mtline.at(mtindex) ;  
}


