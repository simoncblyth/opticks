#pragma once
/**
stree.h : explore minimal approach to geometry translation
============================================================

See also u4/U4Tree.h that creates the stree from Geant4 volumes. 

* this is seeking to replace lots of GGeo code, notably: GInstancer.cc GNode.cc 

TODO:

* controlling the order of repeats with same freq 
* ridx labelling the tree
* maintain correspondence between source nodes and destination nodes thru the factorization
* transform combination   
* transform rebase

**/

#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "NP.hh"

#include "snode.h"
#include "sdigest.h"
#include "sfreq.h"
#include "sstr.h"



struct stree
{
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

    stree();

    std::string desc() const ;
    std::string desc_vec() const ;
    std::string desc_sub() const ;
    std::string desc_sub(const char* sub) const ;

    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr );
    static std::string Digest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ;
    void get_progeny( std::vector<int>& progeny, int nidx ) const ;

    sfreq* make_progeny_freq(int nidx) const ; 
    sfreq* make_freq(const std::vector<int>& nodes ) const ; 

    int find_lvid(const char* soname_, bool starting=true  ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, const char* soname_, bool starting=true ) const ; 
    void find_lvid_nodes( std::vector<int>& nodes, int lvid ) const ; 
    int find_lvid_node( const char* q_soname, int ordinal ) const ; 

    const char* get_soname(int nidx) const ; 
    const char* get_sub(int nidx) const ; 

    int get_parent(int nidx) const ; 
    const glm::tmat4x4<double>& get_transform(int nidx) const ; 

    void get_ancestors(  std::vector<int>& ancestors, int nidx ) const ;
    void get_transform_product( glm::tmat4x4<double>& transform, int nidx ) const ; 

    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ; 

    void classifySubtrees();
    void sortSubtrees(); 


    bool is_contained_repeat(const char* sub) const ; 
    void disqualifyContainedRepeats();

    std::string subtree_digest( int nidx ) const ;
    std::string desc_node(int nidx, bool show_sub=false) const ;
    std::string desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems=10) const ;
    std::string desc_ancestry(int nidx, bool show_sub=false) const ;

    void save( const char* fold ) const ;
    void load( const char* fold );

    static int Compare( const std::vector<int>& a, const std::vector<int>& b ) ; 
    static std::string Desc(const std::vector<int>& a, unsigned edgeitems=10 ) ; 

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



inline std::string stree::desc_sub() const
{
    unsigned num = subs_freq->get_num();
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = subs_freq->get_key(i);
        int freq = subs_freq->get_freq(i);   // -ve freq means disqualified 
        if(freq < FREQ_CUT) continue ;
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

inline void stree::find_lvid_nodes( std::vector<int>& nodes, const char* q_soname, bool starting ) const 
{
    int lvid = find_lvid(q_soname, starting); 
    find_lvid_nodes(nodes, lvid ); 
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

inline int stree::find_lvid_node( const char* q_soname, int ordinal ) const 
{
    std::vector<int> nodes ; 
    find_lvid_nodes(nodes, q_soname ); 
    return ordinal > -1 && ordinal < nodes.size() ? nodes[ordinal] : -1 ; 
}




inline const char* stree::get_soname(int nidx) const
{
    return nidx > -1 ? soname[nds[nidx].lvid].c_str() : "?" ;
}
inline const char* stree::get_sub(int nidx) const 
{
    return nidx > -1 ? subs[nidx].c_str() : nullptr ; 
}
inline int stree::get_parent(int nidx) const 
{ 
    return nidx > -1 ? nds[nidx].parent : -1 ; 
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


/**
stree::is_contained_repeat
--------------------------

Current criteria for a contained repeat is that the 
parent of the first node with the supplied subtree digest 
has a subtree digest frequency equal to that of the 
original first node. 

Notice that this assumes the first node is representative. 

**/

inline bool stree::is_contained_repeat(const char* sub) const
{
    int n_freq = subs_freq->get_freq(sub) ; 
    int nidx = get_first(sub);    // first node with this subtree digest  

    int parent = get_parent(nidx); 
    const char* psub = get_sub(parent) ; 
    int p_freq = subs_freq->get_freq(psub) ; 

    return p_freq == n_freq ; 
}

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

        // disqualification not done during the loop
        // as that would introduce ordering dependency 
    }

    subs_freq->set_disqualify( disqualify ); 

    std::cout 
        << "] stree::disqualifyContainedRepeats " 
        << " disqualify.size " << disqualify.size()
        << std::endl 
        ;
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

inline std::string stree::desc_node(int nidx, bool show_sub) const
{
    const snode& nd = nds[nidx];
    const char* sub = subs[nidx].c_str();
    assert( nd.index == nidx );
    std::stringstream ss ;
    ss << nd.desc() ;
    if(show_sub) ss << " " << subs_freq->desc(sub) ;
    ss << " " << soname[nd.lvid]  ;
    std::string s = ss.str();
    return s ;
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


inline void stree::sortSubtrees()
{
    std::cout << "[ stree::sortSubtrees " << std::endl ;
    stree_subs_freq_ordering ordering(this) ;  

    sfreq::VSU& vsu = subs_freq->vsu ; 
    std::sort( vsu.begin(), vsu.end(), ordering );

    std::cout << "] stree::sortSubtrees " << std::endl ;
}






