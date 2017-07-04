#include "PLOG.hh"

#include "OpticksCSG.h"

#include "NGLMExt.hpp"
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

void NNodeDump::dump(const char* msg) const 
{
    m_node.dump_label("du", msg ) ; 

    bool prim = m_node.is_primitive();

    std::cout 
          << m_node.desc() 
          << ( prim ? " PRIM " : " OPER " )
          << " v:" << std::setw(1) << m_node.verbosity 
          << " " 
          ; 

    nbbox bb = m_node.bbox();
    std::cout << " bb " << bb.desc() << std::endl ; 

    if(!prim)
    {
        std::cout << std::endl ; 
        m_node.left->dump(NULL);
        m_node.right->dump(NULL);
    }
}


void NNodeDump::dump_gtransform( const char* msg) const 
{
    m_node.dump_label("gt", msg) ; 

    if(m_node.gtransform)
    {
        std::cout << gpresent("gtr.t", m_node.gtransform->t ) << std::endl ; 
    }
    else
    {
        std::cout << " NO gtransform " << std::endl ; 
    }

    if(m_node.left && m_node.right)
    {
        m_node.left->dump_gtransform(NULL);
        m_node.right->dump_gtransform(NULL);
    }
}

void NNodeDump::dump_transform( const char* msg) const 
{
    m_node.dump_label("tr", msg) ; 

    if(m_node.transform)
    {
        std::cout << gpresent("tr.t", m_node.transform->t ) << std::endl ; 
    }
    else
    {
        std::cout << " NO transform " << std::endl ; 
    }

    if(m_node.left && m_node.right)
    {
        m_node.left->dump_transform(NULL);
        m_node.right->dump_transform(NULL);
    }
}



void NNodeDump::dump_prim( const char* msg) const 
{
    m_node.dump_label("pr", msg) ; 

    std::vector<const nnode*> prim ;
    m_node.collect_prim(prim);   

    unsigned num_prim = prim.size();
    std::cout << " nprim " << num_prim ; 

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
                   LOG(fatal) << "nnode::dump_prim unhanded shape type " << p->type << " name " << CSGName(p->type) ;
                   assert(0) ;
            }
        }
    }
}


void NNodeDump::dump_planes( const char* msg) const 
{
    m_node.dump_label("pl", msg) ; 
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

