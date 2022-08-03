#pragma once
/**
stree.h : explore minimal approach to geometry translation
============================================================

See also u4/U4Tree.h that creates the stree from Geant4 volumes. 

* this is seeking to replace lots of GGeo code, notably: GInstancer.cc GNode.cc 


* DONE : controlling the order of repeats with same freq, using 2-level sort  


TODO:

* ridx labelling the tree, srepeat summary instances
* maintain correspondence between source nodes and destination nodes thru the factorization
* triplet_identity 
* transform combination   
* transform rebase


mapping from "factorized" instances back to origin PV and vice-versa 
-----------------------------------------------------------------------

Excluding the remainder, the instances each correspond to a contiguous ranges of nidx.
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

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "NP.hh"

#include "snode.h"
#include "sdigest.h"
#include "sfreq.h"
#include "sstr.h"
#include "strid.h"



struct stree
{
    static constexpr const int MAXDEPTH = 15 ; // presentational only   
    static constexpr const int FREQ_CUT = 500 ;  
    // subtree digests with less repeats than FREQ_CUT within the entire geometry 
    // are not regarded as repeats for instancing factorization purposes 

    static constexpr const char* NDS = "nds.npy" ;
    static constexpr const char* TRS = "trs.npy" ;
    static constexpr const char* SONAME = "soname.txt" ;
    static constexpr const char* DIGS = "digs.txt" ;
    static constexpr const char* SUBS = "subs.txt" ;
    static constexpr const char* SUBS_FREQ = "subs_freq" ;

    std::vector<std::string> soname ;
    std::vector<glm::tmat4x4<double>> trs ;
    std::vector<snode> nds ;
    std::vector<std::string> digs ; // single node digest  
    std::vector<std::string> subs ; // subtree digest 
    sfreq* subs_freq ;


    std::vector<glm::tmat4x4<double>> inst ; 
    std::vector<glm::tmat4x4<float>>  inst_f4 ; 


    stree();

    std::string desc() const ;
    std::string desc_vec() const ;
    std::string desc_sub(bool all=false) const ;
    std::string desc_sub(const char* sub) const ;

    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr );
    static std::string Digest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ;
    void get_progeny( std::vector<int>& progeny, int nidx ) const ;
    std::string desc_progeny(int nidx) const ; 


    sfreq* make_progeny_freq(int nidx) const ; 
    sfreq* make_freq(const std::vector<int>& nodes ) const ; 

    int  find_lvid(const char* soname_, bool starting=true  ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, int lvid ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, const char* soname_, bool starting=true ) const ; 
    int  find_lvid_node( const char* q_soname, int ordinal ) const ; 
    int  find_lvid_node( const char* q_spec ) const ; // eg HamamatsuR12860sMask_virtual:0:1000

    const char* get_soname(int nidx) const ; 
    const char* get_sub(   int nidx) const ; 
    int         get_depth( int nidx) const ; 
    int         get_parent(int nidx) const ; 
    int         get_lvid(  int nidx) const ; 
    int         get_copyno(int nidx) const ; 
    const glm::tmat4x4<double>& get_transform(int nidx) const ; 
    void get_ancestors(  std::vector<int>& ancestors, int nidx ) const ;
    void get_transform_product( glm::tmat4x4<double>& transform, int nidx ) const ; 

    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ; 


    std::string subtree_digest( int nidx ) const ;
    static std::string depth_spacer(int depth); 

    std::string desc_node_(int nidx, const sfreq* sf ) const ;
    std::string desc_node(int nidx, bool show_sub=false) const ; 

    std::string desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems=10) const ;
    std::string desc_ancestry(int nidx, bool show_sub=false) const ;

    void save( const char* fold ) const ;
    void load( const char* fold );

    static int Compare( const std::vector<int>& a, const std::vector<int>& b ) ; 
    static std::string Desc(const std::vector<int>& a, unsigned edgeitems=10 ) ; 


    void classifySubtrees();
    bool is_contained_repeat(const char* sub) const ; 
    void disqualifyContainedRepeats();
    void sortSubtrees(); 
    void factorize(); 

    void add_inst( glm::tmat4x4<double>& tr, unsigned gas_idx ); 
    void add_inst(); 
    void save_inst(const char* fold) const ; 
};



inline stree::stree()
    :
    subs_freq(new sfreq)
{
}

inline std::string stree::desc() const
{
    std::stringstream ss ;
    ss << "stree::desc"
       << " nds " << nds.size()
       << " trs " << trs.size()
       << " digs " << digs.size()
       << " subs " << subs.size()
       << " soname " << soname.size()
       << " subs_freq " << std::endl
       << ( subs_freq ? subs_freq->desc() : "-" )
       << std::endl
       ;

    std::string s = ss.str();
    return s ;
}


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
            << desc_node_(nix, sf ) 
            << std::endl
            ; 
    }


    std::string s = ss.str(); 
    return s; 
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

    return ordinal > -1 && ordinal < nodes.size() ? nodes[ordinal] : -1 ; 
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

inline const glm::tmat4x4<double>& stree::get_transform(int nidx) const 
{
    assert( nidx > -1 && nidx < trs.size()); 
    return trs[nidx] ; 
}

inline void stree::get_ancestors( std::vector<int>& ancestors, int nidx ) const
{
    int parent = get_parent(nidx) ;
    while( parent > -1 )
    {
        ancestors.push_back(parent);
        parent = get_parent(parent);
    }
    std::reverse( ancestors.begin(), ancestors.end() );
}

inline void stree::get_transform_product( glm::tmat4x4<double>& transform, int nidx ) const 
{
    std::vector<int> nodes ; 
    get_ancestors(nodes, nidx); 
    nodes.push_back(nidx); 

    glm::tmat4x4<double> xform(1.); 
    for(unsigned i=0 ; i < nodes.size() ; i++ ) 
    {
        int idx = nodes[i] ; 
        const glm::tmat4x4<double>& t = get_transform(idx) ; 
        xform *= t ; 
    }
    assert( sizeof(glm::tmat4x4<double>) == sizeof(double)*16 ); 
    memcpy( glm::value_ptr(transform), glm::value_ptr(xform), sizeof(glm::tmat4x4<double>) );
}




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

inline void stree::save( const char* fold ) const 
{
    NP::Write<int>(    fold, NDS, (int*)nds.data(),    nds.size(), snode::NV );
    NP::Write<double>( fold, TRS, (double*)trs.data(), trs.size(), 4, 4 );
    NP::WriteNames( fold, SONAME, soname );
    NP::WriteNames( fold, DIGS,   digs );
    NP::WriteNames( fold, SUBS,   subs );
    if(subs_freq) subs_freq->save(fold, SUBS_FREQ);
}

inline void stree::load( const char* fold )
{
    NP* a_nds = NP::Load(fold, NDS);
    nds.resize(a_nds->shape[0]);
    memcpy( (int*)nds.data(),    a_nds->cvalues<int>() ,    a_nds->arr_bytes() );

    NP* a_trs = NP::Load(fold, TRS);
    trs.resize(a_trs->shape[0]);
    memcpy( (double*)trs.data(), a_trs->cvalues<double>() , a_trs->arr_bytes() );

    NP::ReadNames( fold, SONAME, soname );
    NP::ReadNames( fold, DIGS,   digs );
    NP::ReadNames( fold, SUBS,   subs );

    if(subs_freq) subs_freq->load(fold, SUBS_FREQ) ;
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

inline void stree::classifySubtrees()
{
    std::cout << "[ stree::classifySubtrees " << std::endl ;
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        std::string sub = subtree_digest(nidx) ;
        subs.push_back(sub) ;
        subs_freq->add(sub.c_str());
    }
    std::cout << "] stree::classifySubtrees " << std::endl ;
}


/**
stree::is_contained_repeat
----------------------------

Original criteria for a contained repeat is that the 
parent of the first node with the supplied subtree digest 
has a subtree digest frequency equal to that of the 
original first node.  

That fails to match GGeo due to a repeats 
inside repeats recursively. 


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

This assumes the first node is representative, might not always be the case. 
But note that the sf digest counts are for entire geometry, not the subtree of one node. 

The deepest sBar is part of 6 potential repeated instances::

    sWall,sPlane,sPanel,sPanelTape,sBar0x71a9200,sBar0x71a9370.

GGeo picked sPanel (due to repeat candidate cut of 500).

* For now I need to duplicate that here.  

**/

inline bool stree::is_contained_repeat(const char* sub) const
{
    int n_freq = subs_freq->get_freq(sub) ; 
    int nidx = get_first(sub);    // first node with this subtree digest  

    int parent = get_parent(nidx); 
    const char* psub = get_sub(parent) ; 
    int p_freq = subs_freq->get_freq(psub) ; 

    return p_freq >= FREQ_CUT ; 
    //return p_freq == n_freq ; 
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
    std::cout << "[ stree::disqualifyContainedRepeats " << std::endl ;

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

    std::cout 
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

inline void stree::sortSubtrees()  // hmm sortSubtreeDigestFreq would be more accurate 
{
    std::cout << "[ stree::sortSubtrees " << std::endl ;
    stree_subs_freq_ordering ordering(this) ;  

    sfreq::VSU& vsu = subs_freq->vsu ; 
    std::sort( vsu.begin(), vsu.end(), ordering );

    std::cout << "] stree::sortSubtrees " << std::endl ;
}

inline void stree::factorize()
{
    std::cout << "[ stree::factorize " << std::endl ;
    classifySubtrees(); 
    disqualifyContainedRepeats();
    sortSubtrees(); 
    std::cout << "] stree::factorize " << std::endl ;
}

inline void stree::add_inst( glm::tmat4x4<double>& tr, unsigned gas_idx )
{
    unsigned ins_idx = inst.size();     // follow sqat4.h::setIdentity
    unsigned ias_idx = 0 ; 

    glm::tvec4<uint64_t> col3 ; 
    col3.x = ins_idx + 1 ; 
    col3.y = gas_idx + 1 ; 
    col3.z = ias_idx + 1 ; 
    col3.w = 0 ; 

    strid::Encode(tr, col3 ); 
    inst.push_back(tr);  
}
inline void stree::add_inst() 
{
    const sfreq* sf = subs_freq ; 

    glm::tmat4x4<double> tr(1.) ; 
    add_inst(tr, 0u );  

    unsigned num = sf->get_num(); 
    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = sf->get_key(i); 
        int freq = sf->get_freq(i); 
        if(freq < FREQ_CUT) continue ; 
        // HMM: dont like continue within such a crucial loop
        // better to have intermediate vector of "srepeat" objects 

        std::vector<int> nodes ; 
        get_nodes(nodes, sub );  

        unsigned gas_idx = i + 1 ; 
        std::cout 
            << " i " << std::setw(3) << i 
            << " gas_idx " << std::setw(3) << gas_idx
            << " sub " << sub
            << " freq " << std::setw(7) << freq
            << " nodes.size " << std::setw(7) << nodes.size()
            << std::endl 
            ;

        assert( int(nodes.size()) == freq );   
        for(unsigned j=0 ; j < nodes.size() ; j++)
        {
            get_transform_product(tr, nodes[j]); 
            add_inst(tr, gas_idx ); 
        }
    }

    strid::Narrow( inst_f4, inst ); 
}

inline void stree::save_inst(const char* fold) const 
{
    NP::Write(fold, "inst.npy",    (double*)inst.data(), inst.size(), 4, 4 ); 
    NP::Write(fold, "inst_f4.npy", (float*)inst_f4.data(), inst_f4.size(), 4, 4 ); 
}




