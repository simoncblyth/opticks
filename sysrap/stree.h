#pragma once
/**
stree.h : explore minimal approach to geometry translation
============================================================

See also u4/U4Tree.h that creates the stree from Geant4 volumes. 

TODO:

* disqualify contained repeats
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


struct stree
{
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
    std::string desc_sub(unsigned freq_cut=100) const ;
    std::string desc_sub(const char* sub) const ;

    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr );
    static std::string Digest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ;
    void get_progeny_r( std::vector<int>& progeny, int nidx ) const ;

    int get_parent(int nidx) const ; 
    const char* get_soname(int nidx) const ; 

    void get_ancestors( std::vector<int>& ancestors, int nidx ) const ;
    void get_nodes(std::vector<int>& nodes, const char* sub) const ;
    void get_depth_range(unsigned& mn, unsigned& mx, const char* sub) const ;
    int get_first( const char* sub ) const ; 


    void classifySubtrees();
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

inline std::string stree::desc_sub(unsigned freq_cut) const
{
    unsigned num = subs_freq->get_num();
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        const char* sub = subs_freq->get_key(i);
        unsigned freq   = subs_freq->get_freq(i);
        if(freq < freq_cut) continue ;
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
       << " " <<  get_soname(first_nidx)
       ;
    std::string s = ss.str();
    return s ; 
}

/**
stree::Digest
----------------------

Progeny digest needs to ncompassing transforms + lvid of subnodes, but only lvid of 
the node in question ?  

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

inline void stree::get_progeny_r( std::vector<int>& progeny , int nidx ) const
{
    std::vector<int> children ;
    get_children(children, nidx);
    std::copy(children.begin(), children.end(), std::back_inserter(progeny));
    for(unsigned i=0 ; i < children.size() ; i++) get_progeny_r(progeny, children[i] );
}

inline int stree::get_parent(int nidx) const { return nidx > -1 ? nds[nidx].parent : -1 ; }
inline const char* stree::get_soname(int nidx) const
{
    return nidx > -1 ? soname[nds[nidx].lvid].c_str() : "?" ;
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

inline void stree::classifySubtrees()
{
    std::cout << "[ stree::classifySubtrees " << std::endl ;
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++)
    {
        std::string sub = subtree_digest(nidx) ;
        subs.push_back(sub) ;
        subs_freq->add(sub.c_str());
    }
    subs_freq->sort();
    std::cout << "] stree::classifySubtrees " << std::endl ;
}

inline std::string stree::subtree_digest(int nidx) const
{
    std::vector<int> progeny ;
    get_progeny_r(progeny, nidx);

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


int stree::Compare( const std::vector<int>& a, const std::vector<int>& b ) // static 
{
    if( a.size() != b.size() ) return -1 ;
    int mismatch = 0 ;
    for(unsigned i=0 ; i < a.size() ; i++) if(a[i] != b[i]) mismatch += 1 ;
    return mismatch ;
}

std::string stree::Desc(const std::vector<int>& a, unsigned edgeitems ) // static 
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









