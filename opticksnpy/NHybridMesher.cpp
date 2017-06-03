#include <sstream>

#include "Timer.hpp"
#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NNode.hpp"

#include "PLOG.hh"


NOpenMesh<NOpenMeshType>* NHybridMesher::make_mesh( const nnode* node, int level , int verbosity, int ctrl )
{
    NOpenMesh<NOpenMeshType>* mesh = NULL ; 

    if(verbosity > 0)
    LOG(info) << "NHybridMesher::make_mesh"
              << " level " << level 
              << " verbosity " << verbosity
              << " ctrl " << ctrl
              ;

    switch(ctrl)
    {
       case 0:
              mesh = new NOpenMesh<NOpenMeshType>(node, level, verbosity, ctrl )  ;
              break ; 
       case 1: 
              mesh = new NOpenMesh<NOpenMeshType>(node, level, verbosity, ctrl )  ;
              mesh->subdiv_test() ;
              break ; 
       case 4: 
              mesh = NOpenMesh<NOpenMeshType>::tetrahedron(level, verbosity, ctrl  ) ; 
              mesh->subdiv_test() ;
              break ; 
       case 6: 
              mesh = NOpenMesh<NOpenMeshType>::cube(level, verbosity, ctrl  ) ; 
              mesh->subdiv_test() ;
              break ; 
       case 666: 
              mesh = NOpenMesh<NOpenMeshType>::hexpatch(level, verbosity, ctrl  ) ; 
              mesh->subdiv_interior_test() ;
              break ; 
    }
    assert(mesh);
    return mesh ; 
}



NHybridMesher::NHybridMesher(const nnode* node, int level , int verbosity, int ctrl)
    :
    m_timer(new Timer),
    m_mesh(make_mesh(node, level, verbosity, ctrl)),
    m_bbox( new nbbox(node->bbox()) ), 
    m_verbosity(verbosity),
    m_ctrl(ctrl)
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






