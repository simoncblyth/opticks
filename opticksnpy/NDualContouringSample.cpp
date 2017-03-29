#include <sstream>

#include "NDualContouringSample.hpp"
#include "NTrianglesNPY.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "DualContouringSample/mesh.h"
#include "DualContouringSample/octree.h"

#include "Timer.hpp"
#include "TimesTable.hpp"

#include "NSphere.hpp"
#include "NNode.hpp"
#include "NBox.hpp"
#include "NFieldCache.hpp"

#include "PLOG.hh"

NDualContouringSample::NDualContouringSample(int level, float threshold, float scale_bb)
  :
   m_timer(new Timer),
   m_level(level),
   m_octreeSize(1 << level),
   m_threshold(threshold),
   m_scale_bb(scale_bb)
{
   m_timer->start();
}

std::string NDualContouringSample::desc()
{
   std::stringstream ss ; 
   ss << "NDualContouringSample"
      << " level " << m_level
      << " octreeSize " << m_octreeSize
      << " threshold " << m_threshold
      << " scale_bb " << m_scale_bb
      ;
   return ss.str(); 
}


void NDualContouringSample::profile(const char* s)
{
   (*m_timer)(s);
}

void NDualContouringSample::report(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << desc() ; 
    TimesTable* tt = m_timer->makeTable();
    tt->dump();
    //tt->save("$TMP");
}



NTrianglesNPY* NDualContouringSample::operator()(nnode* node)
{
    nbbox bb = node->bbox();  // overloaded method 
    std::function<float(float,float,float)> func = node->sdf();

    bb.scale(m_scale_bb);     // kinda assumes centered at origin, slightly enlarge
    bb.side = bb.max - bb.min ; // TODO: see why this not set previously 


    unsigned ctrl = Manager::BUILD_BOTH | Manager::USE_BOTTOM_UP ; 
    //unsigned ctrl = Manager::BUILD_BOTH | Manager::USE_TOP_DOWN ; 
    //unsigned ctrl = Manager::BUILD_BOTTOM_UP | Manager::USE_BOTTOM_UP ; 
    //unsigned ctrl = Manager::BUILD_TOP_DOWN | Manager::USE_TOP_DOWN ; 

    int nominal = m_level ; 
    int coarse  = m_level ; 

    Manager mgr(ctrl, nominal, coarse, m_threshold, &func, bb, m_timer);



    VertexBuffer vertices;
    IndexBuffer indices;

    mgr.buildOctree();
    mgr.generateMeshFromOctree(vertices, indices);

    LOG(info) << " vertices " << vertices.size() ;
    LOG(info) << " indices  " << indices.size() ;

    unsigned npol = indices.size() ; 

    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    IndexBuffer::iterator  pmin = std::min_element(std::begin(indices), std::end(indices));
    IndexBuffer::iterator  pmax = std::max_element(std::begin(indices), std::end(indices));

    size_t imin = std::distance(std::begin(indices), pmin) ;
    size_t imax = std::distance(std::begin(indices), pmax) ;

    LOG(debug) << "min element at: " << imin << " " << indices[imin] ; 
    LOG(debug) << "max element at: " << imax << " " << indices[imax] ;


    profile("_CollectTriangles");
    NTrianglesNPY* tris = new NTrianglesNPY();
    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = indices[t*3 + 0];
         unsigned i1 = indices[t*3 + 1];
         unsigned i2 = indices[t*3 + 2];
          
         MeshVertex& v0 = vertices[i0] ;
         MeshVertex& v1 = vertices[i1] ;
         MeshVertex& v2 = vertices[i2] ;

         tris->add( v0.xyz, v1.xyz, v2.xyz );
         tris->addNormal( v0.normal, v1.normal, v2.normal );
    }
    profile("CollectTriangles");

    report("NDualContouringSample::");

    return tris ; 
}

