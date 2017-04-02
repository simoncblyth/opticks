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






struct sphere_functor 
{
   sphere_functor( float x, float y, float z, float r, bool absolute)
       :   
       center(x,y,z),
       radius(r),
       absolute(absolute)
   {   
   }   

   float operator()( float x, float y, float z) const
   {   
       glm::vec3 p(x,y,z) ;
       float d = glm::distance( p, center );
       float v = d - radius ; 
       return absolute ? fabs(v) : v  ;
   }   

   std::string desc();


   glm::vec3 center ; 
   float     radius ; 
   bool      absolute ; 

};







NImplicitMesher::NImplicitMesher(int resolution, int verbosity, float scale_bb, int ctrl)
  :
   m_timer(new Timer),
   m_resolution(resolution),
   m_verbosity(verbosity),
   m_scale_bb(scale_bb),  
   m_ctrl(ctrl)
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
      << " ctrl " << m_ctrl
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
    if(m_ctrl == 10)
    {
        LOG(warning) << "NImplicitMesher::operator() ctrl override return sphere " ; 
        return sphere_test(); 
    }

    nbbox bb = node->bbox(); 
    std::function<float(float,float,float)> sdf = node->sdf();

    bb.scale(m_scale_bb);     // kinda assumes centered at origin, slightly enlarge
    bb.side = bb.max - bb.min ;

    glm::vec3 min(bb.min.x, bb.min.y, bb.min.z );
    glm::vec3 max(bb.max.x, bb.max.y, bb.max.z );

    float tval = 0.5f ; 
    float absolute = true ; 
    ImplicitMesherF im(sdf, tval, absolute ); 

    im.setParam(m_resolution, min, max);
    im.polygonize();
    im.dump();
    
    const std::vector<glm::vec3>& verts = im.vertices();
    const std::vector<glm::vec3>& norms = im.normals();
    const std::vector<glm::ivec3>& tris = im.triangles();

    NTrianglesNPY* tt = collectTriangles( verts, norms, tris );

    report("NImplicitMesher::");

    return tt ; 
}


NTrianglesNPY* NImplicitMesher::sphere_test()
{
    glm::vec3 min(-10,-10,-10);
    glm::vec3 max( 10, 10, 10);
    float tval = 0.5f ; 
    float absolute = true ; 

    sphere_functor ssf(0,0,0,10, false);   // false:signed SDF, absolution done in ImplicitFunction

    std::function<float(float,float,float)> sfn = ssf ; 

    ImplicitMesherF im(sfn, tval, absolute); 
    im.setParam(m_resolution, min, max);
    im.polygonize();
    im.dump();

    const std::vector<glm::vec3>& verts = im.vertices();
    const std::vector<glm::vec3>& norms = im.normals();
    const std::vector<glm::ivec3>& tris = im.triangles();

    NTrianglesNPY* tt = collectTriangles( verts, norms, tris );
 
    return tt ; 
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








