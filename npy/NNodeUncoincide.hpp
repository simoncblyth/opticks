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

#include <glm/fwd.hpp>
#include "NNodeEnum.hpp"
#include "NPY_API_EXPORT.hh"



struct nuv ; 
struct nnode ; 
struct NNodeNudger ; 

/*
NNodeUncoincide
=================

* this is not enanbled in NCSG, but NNodeNudger is  


See issues/NScanTest_csg_zero_crossings.rst 

* pairwise uncoincidence is not helping much, 
  need to act at tree level 

Canonically invoked after CSG import 
via NCSG::postimport_uncoincide/nnode::uncoincide


Hmm nothing fancy needed to see the coincidence,
equality of a.bbox.min.z and b.bbox.max.z or vv.
    
Shapes with separate +z and -z parameters are
easy to nudge in +z, -z direction.  

* CSG_CYLINDER
* CSG_CONE
* CSG_ZSPHERE

Shapes with symmetric parameters like box3 are a pain, as 
to grow in eg +z direction need to grow in both +z and -z
first and then transform to keep the other side at same place.

Hmm to avoid this perhaps make a CSG_ZBOX primitive ? 


*/

class NPY_API NNodeUncoincide
{
    public:
        NNodeUncoincide(nnode* node, float epsilon, unsigned verbosity );
    public:
        unsigned uncoincide();

    private:
        void init();
        // treewise approach 
        unsigned uncoincide_treewise();
        unsigned uncoincide_treewise_fiddle();
        unsigned uncoincide_uncyco(nnode* node);
    private:
        // pairwise approach 
        bool     is_uncoincidable_subtraction(nnode*& a, nnode*& b) ;
        bool     is_uncoincidable_union(nnode*& a, nnode*& b) ;
        unsigned uncoincide_subtraction(nnode* a, nnode* b) ;
        unsigned uncoincide_union(nnode* a, nnode* b) ;

    private:
        nnode*       m_node ; 
        float        m_epsilon ; 
        int          m_verbosity ; 
        NNodeNudger* m_nudger ;   // NB are rejiging to put another nudger directly in NCSG

};


