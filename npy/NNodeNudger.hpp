/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <vector>

#include "NNodeEnum.hpp"
#include "NNodeCoincidence.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

#include "plog/Severity.h"



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


template <typename T> class  NPY ; 


struct NPY_API NNodeNudger 
{
    static std::vector<unsigned>*  TreeList ;  
    static NPY<unsigned>* NudgeBuffer ; 
    static void SaveBuffer(const char* path) ; 

    nnode* root ; 
    const float epsilon ; 
    const unsigned verbosity ; 
    bool listed ; 
    bool enabled ; 
    plog::Severity level ; 


    std::vector<nnode*>       prim ; 
    std::vector<nbbox>        bb ; 
    std::vector<nbbox>        cc ; 
    std::vector<unsigned>     zorder ; 
    std::vector<NNodeCoincidence> coincidence ; 
    std::vector<NNodeCoincidence> nudges ; 

    NNodeNudger(nnode* root, float epsilon, unsigned verbosity) ;
  
    void init();
    void update_prim_bb();  // direct from param, often with gtransform applied
    bool operator()( int i, int j)  ;


    void uncoincide();

    void collect_coincidence();
    void collect_coincidence(unsigned i, unsigned j);
    unsigned get_num_coincidence() const ; 
    unsigned get_num_prim() const ; 
    std::string desc_coincidence() const ;
    std::string brief() const ;

    bool can_znudge(const NNodeCoincidence* coin) const ;
    void znudge(NNodeCoincidence* coin);

    bool can_znudge_union_maxmin(const NNodeCoincidence* coin) const ;
    void znudge_union_maxmin(NNodeCoincidence* coin);

    bool can_znudge_difference_minmin(const NNodeCoincidence* coin) const ;
    void znudge_difference_minmin(NNodeCoincidence* coin);


    void dump(const char* msg="NNodeNudger::dump");
    void dump_qty(char qty, int wid=10);
    void dump_joins();

};

// end of NNodeNudger 



/*

   * know how to handle siblings of union parent
     with minmax or maxmin pair coincidence

   * difference coincidence will often be non-siblings, eg 
     (cy-cy)-co when the base of the subtracted cone lines up with 
      the first cylinder ... perhaps should +ve-ize 


            -
         -    co
       cy cy

     +ve form:

            *
         *    !co
       cy !cy


     

     Consider (cy - co) with coincident base...
     solution is to grow co down, but how to 
     detect in code ? 

     When you get minmin coincidence ~~~
     (min means low edge... so direction to grow
      is clear ? Check parents of the pair and
      operate on one with the "difference" parent, 
      ie the one being subtracted) 

     Nope they could both be being subtracted ?


     A minmin coincidence after positivization, 
     can always pull down the one with the complement ?



                        +-----+
                       /       \
                      /         \
             +-------*-------+   \
             |      /        |    \
             |     /         |     \
             |    /          |      \
             |   /           |       \
             |  /            |        \
             | /             |         \
             |/              |          \
             *               |           \
            /|               |            \
           / |               |        B    \
          /  |  A            |              \
         /   |               |               \
        /    |               |                \
       +-----+~~~~~~~~~~~~~~~+-----------------+


*/


