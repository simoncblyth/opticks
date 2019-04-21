#include <sstream>

#include "BFile.hh"

#include "NHybridMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NOpenMesh.hpp"
#include "NParameters.hpp"
#include "NNode.hpp"

#include "PLOG.hh"



NHybridMesher::NHybridMesher(const nnode* node, NParameters* meta, const char* treedir)
    :
    m_mesh(MESH::Make(node, meta, treedir)),
    m_bbox( new nbbox(node->bbox()) ), 
    m_treedir(treedir ? strdup(treedir) : NULL )
{
}


std::string NHybridMesher::desc()
{
   std::stringstream ss ; 
   ss << "NHybridMesher"
      ;
   return ss.str(); 
}


NTrianglesNPY* NHybridMesher::operator()()
{
    NTrianglesNPY* tt = new NTrianglesNPY(m_mesh);  // NTriSource pull out the tris
    return tt ; 
}





