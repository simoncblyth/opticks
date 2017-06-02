#include <sstream>

#include "Timer.hpp"
#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NNode.hpp"

#include "PLOG.hh"


NHybridMesher::NHybridMesher(const nnode* node, int level , int verbosity, int ctrl)
    :
    m_timer(new Timer),
    m_mesh(new NOpenMesh<NOpenMeshType>(node, level, verbosity, ctrl)),
    m_bbox( new nbbox(node->bbox()) ), 
    m_verbosity(verbosity)
{
}


std::string NHybridMesher::desc()
{
   std::stringstream ss ; 
   ss << "NHybridMesher"
      << " verbosity " << m_verbosity
      ;
   return ss.str(); 
}


NTrianglesNPY* NHybridMesher::operator()()
{
    NTrianglesNPY* tt = new NTrianglesNPY(m_mesh);  // NTriSource pull out the tris
    return tt ; 
}






