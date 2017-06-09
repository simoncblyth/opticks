#include <sstream>

#include "BFile.hh"

#include "Timer.hpp"
#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NNode.hpp"

#include "PLOG.hh"


NOpenMesh<NOpenMeshType>* NHybridMesher::make_mesh( const nnode* node, int level , int verbosity, int ctrl, NPolyMode_t polymode, const char* polycfg, const char* treedir)
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
         mesh = new MESH(node, level, verbosity, ctrl, polycfg, meshmode )  ;
         assert(mesh);
    }
    else
    {
        mesh = MESH::BuildTest( level, verbosity, ctrl, polycfg );
        assert(mesh);
        mesh->subdiv_test() ;
    }


     if(treedir && mesh->cfg.offsave > 0) 
     {
         std::string meshpath = BFile::FormPath(treedir, "mesh.off") ;
         LOG(info) << "cfg.offsave writing mesh to " << meshpath ; 
         mesh->write(meshpath.c_str());
     }


    return mesh ; 
}


NHybridMesher::NHybridMesher(const nnode* node, int level , int verbosity, int ctrl, NPolyMode_t polymode, const char* polycfg, const char* treedir )
    :
    m_timer(new Timer),
    m_mesh(make_mesh(node, level, verbosity, ctrl, polymode, polycfg, treedir)),
    m_bbox( new nbbox(node->bbox()) ), 
    m_verbosity(verbosity),
    m_ctrl(ctrl),
    m_polycfg(polycfg ? strdup(polycfg) : NULL ),
    m_treedir(treedir ? strdup(treedir) : NULL )
{
}



std::string NHybridMesher::desc()
{
   std::stringstream ss ; 
   ss << "NHybridMesher"
      << " verbosity " << m_verbosity
      << " polycfg " << ( m_polycfg ? m_polycfg : "-" )
      ;
   return ss.str(); 
}


NTrianglesNPY* NHybridMesher::operator()()
{
    NTrianglesNPY* tt = new NTrianglesNPY(m_mesh);  // NTriSource pull out the tris
    return tt ; 
}





