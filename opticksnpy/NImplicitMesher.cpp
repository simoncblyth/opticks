#include <sstream>

#include "ImplicitMesher/ImplicitMesherF.h"

#include "BStr.hh"

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




NImplicitMesher::NImplicitMesher(nnode* node, int resolution, int verbosity, float scale_bb, int ctrl, std::string seedstr)
    :
    m_timer(new Timer),
    m_node(node),
    m_bbox( new nbbox(node->bbox()) ), 
    m_sdf( node->sdf() ),
    m_mesher(NULL),
    m_resolution(resolution),
    m_verbosity(verbosity),
    m_scale_bb(scale_bb),  
    m_ctrl(ctrl),
    m_seedstr(seedstr)
{
    init();
}


std::string NImplicitMesher::desc()
{
   std::stringstream ss ; 
   ss << "NImplicitMesher"
      << " resolution " << m_resolution
      << " verbosity " << m_verbosity
      << " scale_bb " << m_scale_bb
      << " ctrl " << m_ctrl
      << " seedstr " << m_seedstr
      ;
   return ss.str(); 
}

void NImplicitMesher::init()
{
    m_timer->start();

    m_bbox->scale(m_scale_bb);  // kinda assumes centered at origin, slightly enlarge
    m_bbox->side = m_bbox->max - m_bbox->min ;

    float tval = 0.f ; 
    float negate = false ; 
    m_mesher = new ImplicitMesherF(m_sdf, m_verbosity, tval, negate ); 

    glm::vec3 min(m_bbox->min.x, m_bbox->min.y, m_bbox->min.z );
    glm::vec3 max(m_bbox->max.x, m_bbox->max.y, m_bbox->max.z );

    m_mesher->setParam(m_resolution, min, max);

    addManualSeeds();
    addCenterSeeds();

}

void NImplicitMesher::addManualSeeds()
{
    std::vector<float> seed ; 
    if(!m_seedstr.empty()) BStr::fsplit(seed, m_seedstr.c_str(), ',');

    unsigned nseed = seed.size();
    if(nseed > 0)
    {
        if(nseed % 3 == 0)
        {
            for(unsigned i=0 ; i < nseed/3 ; i++ )
            {
               float sx = seed[i*3+0]; 
               float sy = seed[i*3+1]; 
               float sz = seed[i*3+2];
               LOG(info) << "NImplicitMesher::addManualSeeds nseed " << nseed << " addSeed (" << sx << " " << sy << " " << sz << ") " ; 
               m_mesher->addSeed(sx, sy, sz); 
            }
        }
        else
        {
            LOG(warning) << "NImplicitMesher::addManualSeeds ignoring seeds as not a multiple of 3 for x,y,z coordinates : " << nseed ; 
        }
    } 
}

void NImplicitMesher::addCenterSeeds()
{
    std::vector<glm::vec3> centers ; 
    m_node->collect_prim_centers(centers);

    unsigned ncenters = centers.size();
    for(unsigned i=0 ; i < ncenters ; i++)
    {
        const glm::vec3& c = centers[i] ; 
        LOG(info) << "NImplicitMesher::addCenterSeeds " << i << "/" << ncenters << " addSeed (" << c.x << "," << c.y << "," << c.z << ")" ; 
        m_mesher->addSeed(c.x, c.y, c.z );
    }
}


NTrianglesNPY* NImplicitMesher::operator()()
{
    LOG(info) << "NImplicitMesher::operator() bb " << m_bbox->desc() ; 

    m_mesher->polygonize();
    m_mesher->dump();
    
    const std::vector<glm::vec3>& verts = m_mesher->vertices();
    const std::vector<glm::vec3>& norms = m_mesher->normals();
    const std::vector<glm::ivec3>& tris = m_mesher->triangles();

    NTrianglesNPY* tt = collectTriangles( verts, norms, tris );

    report("NImplicitMesher::");

    return tt ; 
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




NTrianglesNPY* NImplicitMesher::collectTriangles(const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::ivec3>& tris )
{
    int ntri = tris.size();
    int nvert = verts.size();
    assert( verts.size() == norms.size() );

    NTrianglesNPY* tt = new NTrianglesNPY();

    for(int t=0 ; t < ntri ; t++)
    {
         const glm::ivec3& tri = tris[t] ;   

         int i0 = tri.x ;
         int i1 = tri.y ;
         int i2 = tri.z ;

         assert( i0 < nvert && i1 < nvert && i2 < nvert );
          
         const glm::vec3& v0 = verts[i0] ;
         const glm::vec3& v1 = verts[i1] ;
         const glm::vec3& v2 = verts[i2] ;

         const glm::vec3& n0 = norms[i0] ;
         const glm::vec3& n1 = norms[i1] ;
         const glm::vec3& n2 = norms[i2] ;

         tt->add( v0, v1, v2 );
         tt->addNormal( n0, n1, n2 );
    }
    m_timer->stamp("CollectTriangles");
    return tt ; 
}







