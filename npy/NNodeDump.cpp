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

#include "PLOG.hh"

#include "OpticksCSG.h"

#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "GLMFormat.hpp"
#include "NNode.hpp"
#include "NBBox.hpp"
#include "NNodeDump.hpp"

#include "NPrimitives.hpp"

NNodeDump::NNodeDump(const nnode& node)
   :
   m_node(node)
{
}


void NNodeDump::dump() const 
{
    dump_base();
    dump_gtransform();

    if(m_node.verbosity > 1)
    {
        LOG(info) << "NNodeDump::dump verbosity " << m_node.verbosity ;         
        dump_prim();
        dump_transform();
        dump_planes();
    }
}

void NNodeDump::dump_label(const char* pfx) const 
{
    std::cout 
         << std::setw(3) << (  pfx ? pfx : "-" ) << " " 
         << std::setw(nnode::desc_indent) << m_node.desc() 
         ; 
}

void NNodeDump::dump_base() const 
{
    dump_label("du") ; 

    bool prim = m_node.is_primitive();

    std::cout 
          << ( prim ? " PRIM " : " OPER " )
          << " v:" << std::setw(1) << m_node.verbosity 
          << " " 
          ; 

    nbbox bb = m_node.bbox();
    std::cout << " bb " << bb.desc() << std::endl ; 

    if(!prim)
    {
        std::cout << std::endl ; 
        m_node.left->dump();
        m_node.right->dump();
    }
}



void NNodeDump::dump_gtransform() const 
{
    dump_label("gt") ; 

    if(m_node.gtransform)
    {
        std::cout << gpresent_label("gt.t") << std::endl ; 
        std::cout << gpresent(NULL, m_node.gtransform->t ) << std::endl ; 
    }
    else
    {
        std::cout << " NO gtransform " << std::endl ; 
    }

    if(m_node.left && m_node.right)
    {
        m_node.left->dump_gtransform();
        m_node.right->dump_gtransform();
    }
}

void NNodeDump::dump_transform() const 
{
    dump_label("tr") ; 

    if(m_node.transform)
    {
        std::cout << gpresent_label("tr.t") << std::endl ; 
        std::cout << gpresent(NULL, m_node.transform->t ) << std::endl ; 
    }
    else
    {
        std::cout << " NO transform " << std::endl ; 
    }

    if(m_node.left && m_node.right)
    {
        m_node.left->dump_transform();
        m_node.right->dump_transform();
    }
}



void NNodeDump::dump_prim() const 
{
    dump_label("pr") ; 

    std::vector<const nnode*> prim ;
    m_node.collect_prim(prim);   

    unsigned num_prim = prim.size();
    std::cout << " nprim " << num_prim << std::endl ; 

    for(unsigned i=0 ; i < num_prim ; i++)
    {
        const nnode* p = prim[i] ; 
        switch(p->type)
        {
            case CSG_SPHERE     : ((nsphere*)p)->pdump("sp") ; break ; 
            case CSG_ZSPHERE    : ((nzsphere*)p)->pdump("zs") ; break ; 
            case CSG_BOX        :    ((nbox*)p)->pdump("bx") ; break ; 
            case CSG_BOX3       :    ((nbox*)p)->pdump("bx") ; break ; 
            case CSG_SLAB       :  ((nslab*)p)->pdump("sl") ; break ; 
            case CSG_CYLINDER   :  ((ncylinder*)p)->pdump("cy") ; break ; 
            case CSG_CONE       :  ((ncone*)p)->pdump("co") ; break ; 
            case CSG_DISC       :  ((ndisc*)p)->pdump("dc") ; break ; 
            case CSG_CONVEXPOLYHEDRON  :  ((nconvexpolyhedron*)p)->pdump("cp") ; break ; 

            default:
            {
                   LOG(fatal) << "nnode::dump_prim unhanded shape type " << p->type << " name " << CSG::Name(p->type) ;
                   assert(0) ;
            }
        }
    }
}


void NNodeDump::dump_planes() const 
{
    dump_label("pl") ; 
    unsigned num_planes = m_node.planes.size() ;
    std::cout  
              << " num_planes " << num_planes
              << std::endl 
              ;

    for(unsigned i=0 ; i < num_planes ; i++)
    {
        glm::vec4 pl = m_node.planes[i];
        glm::vec3 normal(pl.x, pl.y, pl.z);
        float dist = pl.w ; 
        std::cout << " i " << std::setw(2) << i 
                  << " pl " << gpresent(pl)
                  << " nlen " << glm::length(normal)
                  << " dist " << dist
                  << std::endl ; 
    }
}



