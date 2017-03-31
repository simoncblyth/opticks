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

#include "NGrid3.hpp"
#include "NField3.hpp"
#include "NFieldGrid3.hpp"
#include "NOctools.hpp"

#include "PLOG.hh"

NDualContouringSample::NDualContouringSample(int nominal, int coarse, int verbosity, float threshold, float scale_bb)
  :
   m_timer(new Timer),
   m_nominal(nominal),
   m_coarse(coarse),
   m_verbosity(verbosity),
   m_threshold(threshold),
   m_scale_bb(scale_bb)
{
   m_timer->start();
}

std::string NDualContouringSample::desc()
{
   std::stringstream ss ; 
   ss << "NDualContouringSample"
      << " nominal " << m_nominal
      << " coarse " << m_coarse
      << " verbosity " << m_verbosity
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


    glm::vec3 bb_min(bb.min.x, bb.min.y, bb.min.z );
    glm::vec3 bb_max(bb.max.x, bb.max.y, bb.max.z );


    //unsigned ctrl = BUILD_BOTH | USE_BOTTOM_UP ; 
    //unsigned ctrl = BUILD_BOTH | USE_TOP_DOWN ; 
    unsigned ctrl = BUILD_BOTTOM_UP | USE_BOTTOM_UP ; 
    //unsigned ctrl = BUILD_TOP_DOWN | USE_TOP_DOWN ; 

    bool offset = true ; // <-- TODO: do the dev to switch this off

    NField<glm::vec3,glm::ivec3,3> field(&func, bb_min, bb_max );

    NGrid<glm::vec3,glm::ivec3,3>  grid(m_nominal);

    NFieldGrid3<glm::vec3,glm::ivec3> fieldgrid(&field, &grid, offset);

    NManager<OctreeNode> mgr(ctrl, m_nominal, m_coarse, m_verbosity, m_threshold, &fieldgrid, bb, m_timer);

    mgr.buildOctree();

    OctCheck raw(mgr.getRaw()) ; 
    raw.report("raw") ;
    assert(raw.ok());


    mgr.simplifyOctree();


    OctCheck simp(mgr.getSimplified());
    if(!simp.ok()) simp.report("simplified") ;
    assert(simp.ok());


    mgr.generateMeshFromOctree();

    NTrianglesNPY* tris = mgr.collectTriangles();

    report("NDualContouringSample::");

    return tris ; 
}

