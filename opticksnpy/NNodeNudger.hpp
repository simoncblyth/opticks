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

    NNodeNudger(nnode* root, float epsilon, unsigned verbosity) ;
  
    void init();
    void update_bb();
    bool operator()( int i, int j)  ;

    void znudge();
    void znudge_anypair();
    void znudge_anypair(unsigned i, unsigned j);
    void znudge_lineup();

    void dump(const char* msg="NNodeNudger::dump");
    void dump_qty(char qty, int wid=10);
    void dump_joins();

};

// end of NNodeNudger 


