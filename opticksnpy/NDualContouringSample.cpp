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

    //unsigned ctrl = BUILD_BOTH | USE_BOTTOM_UP ; 
    //unsigned ctrl = BUILD_BOTH | USE_TOP_DOWN ; 
    unsigned ctrl = BUILD_BOTTOM_UP | USE_BOTTOM_UP ; 
    //unsigned ctrl = BUILD_TOP_DOWN | USE_TOP_DOWN ; 

    NField3 field(&func, bb.min, bb.max );
    NGrid3  grid(m_nominal);

    bool offset = true ; // <-- TODO: do the dev to switch this off
    NFieldGrid3 fieldgrid(&field, &grid, offset);

    NManager<OctreeNode> mgr(ctrl, m_nominal, m_coarse, m_verbosity, m_threshold, &fieldgrid, bb, m_timer);

    mgr.buildOctree();

    mgr.generateMeshFromOctree();

    NTrianglesNPY* tris = mgr.collectTriangles();

    report("NDualContouringSample::");

    return tris ; 
}

