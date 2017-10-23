#pragma once

#include <vector>

#include "NNodeEnum.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

struct nnode ; 


/*
NNodeNudger
============

Requirements for znudge-ability of primitives:

* increase/decrease_z1/z2 controls 
* r2()/r1() methods   TODO: generalization for a new CSG_ZBOX primitive


NB canonical way of invoking this via NCSG::LoadCSG is sensitive to 
VERBOSITY envvar. 
 

::

        +--------------+ .
        |              |
        |           . ++-------------+
        |             ||             |
        |         rb  ||  ra         |
        |             ||             | 
        |           . || .           |    
        |             ||             |
        |             ||          b  |
        |           . ++-------------+
        |  a           |
        |              |
        +--------------+ .

                      za  
                      zb                      

        ------> Z

*/



struct NPY_API NNodeCoincidence 
{
    NNodeCoincidence(nnode* i_, nnode* j_, NNodePairType p_) 
         :
         i(i_),
         j(j_),
         p(p_),
         fixed(false)
    {} ;

    nnode* i ; 
    nnode* j ;
    NNodePairType p ; 
    bool   fixed ; 

    std::string desc() const ; 

    bool is_union_siblings() const;
    bool is_union_parents() const;
    bool is_same_union() const;
    bool is_siblings() const;


};




struct NPY_API NNodeNudger 
{
    nnode* root ; 
    const float epsilon ; 
    const unsigned verbosity ; 
    unsigned znudge_count ; 

    std::vector<nnode*>       prim ; 
    std::vector<nbbox>        bb ; 
    std::vector<nbbox>        cc ; 
    std::vector<unsigned>     zorder ; 
    std::vector<NNodeCoincidence> coincidence ; 

    NNodeNudger(nnode* root, float epsilon, unsigned verbosity) ;
  
    void init();
    void update_prim_bb();  // direct from param, often with gtransform applied
    bool operator()( int i, int j)  ;


    void uncoincide();

    void collect_coincidence();
    void collect_coincidence(unsigned i, unsigned j);
    unsigned get_num_coincidence() const ; 
    std::string desc_coincidence() const ;

    bool can_znudge(const NNodeCoincidence* coin) const ;
    void znudge(NNodeCoincidence* coin);

    bool can_znudge_umaxmin(const NNodeCoincidence* coin) const ;
    void znudge_umaxmin(NNodeCoincidence* coin);

    bool can_znudge_dminmin(const NNodeCoincidence* coin) const ;
    void znudge_dminmin(NNodeCoincidence* coin);




    void dump(const char* msg="NNodeNudger::dump");
    void dump_qty(char qty, int wid=10);
    void dump_joins();

};

// end of NNodeNudger 


