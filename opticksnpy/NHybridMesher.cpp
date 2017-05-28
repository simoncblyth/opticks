#include <sstream>

#include "Timer.hpp"
#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NNode.hpp"

#include "PLOG.hh"


NHybridMesher::NHybridMesher(nnode* node, int level , int verbosity)
    :
    m_timer(new Timer),
    m_node(node),
    m_bbox( new nbbox(node->bbox()) ), 
    m_sdf( node->sdf() ),
    m_level(level),
    m_nu(1 << level),
    m_nv(1 << level),
    m_verbosity(verbosity)
{
}


std::string NHybridMesher::desc()
{
   std::stringstream ss ; 
   ss << "NHybridMesher"
      << " level " << m_level
      << " nu " << m_nu
      << " nv " << m_nv
      << " verbosity " << m_verbosity
      ;
   return ss.str(); 
}


NTrianglesNPY* NHybridMesher::operator()()
{
    NOpenMesh<NOpenMeshType>* mesh = new NOpenMesh<NOpenMeshType>() ;

    mesh->build_parametric( m_node, m_nu, m_nv, m_verbosity  ); 

    m_node->mesh = mesh   ;      

    NTrianglesNPY* tt = new NTrianglesNPY(mesh);  // NTriSource pull out the tris

    return tt ; 
}






