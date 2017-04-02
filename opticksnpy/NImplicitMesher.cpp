#include <sstream>

#include "ImplicitMesher/ImplicitMesherF.h"

#include "NImplicitMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "Timer.hpp"
#include "TimesTable.hpp"

#include "NSphere.hpp"
#include "NNode.hpp"
#include "NBox.hpp"

#include "PLOG.hh"

NImplicitMesher::NImplicitMesher(int resolution, int verbosity, float scale_bb)
  :
   m_timer(new Timer),
   m_resolution(resolution),
   m_verbosity(verbosity),
   m_scale_bb(scale_bb)
{
   m_timer->start();
}

std::string NImplicitMesher::desc()
{
   std::stringstream ss ; 
   ss << "NImplicitMesher"
      << " resolution " << m_resolution
      << " verbosity " << m_verbosity
      << " scale_bb " << m_scale_bb
      ;
   return ss.str(); 
}

void NImplicitMesher::profile(const char* s)
{
   (*m_timer)(s);
}

void NImplicitMesher::report(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << desc() ; 
    TimesTable* tt = m_timer->makeTable();
    tt->dump();
    //tt->save("$TMP");
}


NTrianglesNPY* NImplicitMesher::operator()(nnode* node)
{
    nbbox bb = node->bbox(); 
    std::function<float(float,float,float)> func = node->sdf();

    bb.scale(m_scale_bb);     // kinda assumes centered at origin, slightly enlarge
    bb.side = bb.max - bb.min ;

    glm::vec3 bb_min(bb.min.x, bb.min.y, bb.min.z );
    glm::vec3 bb_max(bb.max.x, bb.max.y, bb.max.z );


    ImplicitMesherF mesher(*node); 

    mesher.setParam(m_resolution, bb_min, bb_max);
    mesher.polygonize();
    mesher.dump();
    

    NTrianglesNPY* tris = NULL ; 

    report("NImplicitMesher::");

    return tris ; 
}

