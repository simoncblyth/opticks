#include <sstream>

#include "Timer.hpp"
#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NNode.hpp"

#include "PLOG.hh"


NOpenMesh<NOpenMeshType>* NHybridMesher::make_mesh( const nnode* node, int level , int verbosity, int ctrl, NPolyMode_t polymode )
{
    assert(polymode == POLY_HY || polymode == POLY_BSP) ;
    NOpenMeshMode_t meshmode = polymode == POLY_HY ? COMBINE_HYBRID : COMBINE_CSGBSP ; 

    NOpenMesh<NOpenMeshType>* mesh = NULL ; 

    typedef NOpenMesh<NOpenMeshType> MESH ; 

    if(verbosity > 0)
    LOG(info) << "NHybridMesher::make_mesh"
              << " level " << level 
              << " verbosity " << verbosity
              << " ctrl " << ctrl
              << " polymode " << polymode
              << " polymode " << NPolygonizer::PolyModeString(polymode)
              << " meshmode " << meshmode
              << " MeshModeString " << MESH::MeshModeString(meshmode) 
              ;

    if(ctrl == 0)
    {
         mesh = new MESH(node, level, verbosity, ctrl, meshmode )  ;
         assert(mesh);
    }
    else
    {
        switch(ctrl)
        {
           case   4: mesh = MESH::tetrahedron(level, verbosity, ctrl  )    ; break ; 
           case   6: mesh = MESH::cube(level, verbosity, ctrl  )           ; break ; 
           case  66: mesh = MESH::hexpatch_inner(level, verbosity, ctrl  ) ; break ; 
           case 666: mesh = MESH::hexpatch(level, verbosity, ctrl  )       ; break ; 
        }
        assert(mesh);
        mesh->subdiv_test() ;
    }
    return mesh ; 
}


NHybridMesher::NHybridMesher(const nnode* node, int level , int verbosity, int ctrl, NPolyMode_t polymode )
    :
    m_timer(new Timer),
    m_mesh(make_mesh(node, level, verbosity, ctrl, polymode)),
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






