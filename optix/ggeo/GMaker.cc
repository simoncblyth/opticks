#include <glm/glm.hpp>
#include "GLMFormat.hpp"

#include "GMaker.hh"

#include "GCache.hh"
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GParts.hh"


#include "GBBoxMesh.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GVector.hh"
#include "GMatrix.hh"

// npy-
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NLog.hpp"

#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include <glm/glm.hpp>



const char* GMaker::SPHERE = "sphere" ; 
const char* GMaker::ZSPHERE = "zsphere" ; 
const char* GMaker::ZSPHEREINTERSECT = "zsphereintersect" ; 
const char* GMaker::BOX = "box" ; 
const char* GMaker::PMT = "pmt" ; 
const char* GMaker::UNDEFINED = "undefined" ; 
 
const char* GMaker::ShapeName(char shapecode)
{
    switch(shapecode) 
    {
       case 'B':return BOX     ; break ; 
       case 'S':return SPHERE  ; break ; 
       case 'Z':return ZSPHERE ; break ; 
       case 'L':return ZSPHEREINTERSECT ; break ; 
       case 'P':return PMT     ; break ; 
       case 'U':return UNDEFINED ; break ;
    }
    return NULL ;
} 




std::vector<GSolid*> GMaker::make(unsigned int index, char shapecode, glm::vec4& param, const char* spec )
{
    GSolid* solid = NULL ; 

    // TODO: split composites from primitives

    std::vector<GSolid*> solids ; 
    switch(shapecode)
    {
        case 'B': 
                  {
                      solid = makeBox(param);
                      solids.push_back(solid); 
                  }
                  break;
        case 'S': 
                  {
                      // I:icosahedron O:octahedron HO:hemi-octahedron C:cube 
                      const char* type = "I" ; 
                      solid = makeSubdivSphere(param, 3, type) ;
                      solids.push_back(solid); 
                  }
                  break;
        case 'Z':
                  {
                      solid = makeZSphere(param) ;
                      solids.push_back(solid); 
                  }
                  break;
        case 'L':
                  makeZSphereIntersect(solids, param, spec) ;
                  break;

    }

    unsigned int nsolids = solids.size();

    if(nsolids == 1)
    {
        solid = solids[0] ;
        float bbscale = 1.00001f ; 
        GParts* pts = GParts::make(shapecode, param, spec, bbscale);
        solid->setParts(pts);
    }
    else
    {
        // composite analytics must be handled at lower level 
    }


    unsigned int boundary = m_bndlib->addBoundary(spec);
    for(unsigned int i=0 ; i < solids.size() ; i++)
    {
         solid = solids[i];
         solid->setBoundary(boundary);

         //solid->setIndex(index);

         //GParts* pts = solid->getParts(); 
         // moved these to GGeoTest
         //if(pts)
         //{
           //  pts->setIndex(0u, solid->getIndex());
           //  pts->setNodeIndex(0u, solid->getIndex());
           //  pts->setBndLib(m_bndlib);
         //}
    } 

    return solids ; 
}


void GMaker::init()
{
    m_ggeo = m_cache->getGGeo();
    if(m_ggeo)
    {
        m_geolib = m_ggeo->getGeoLib();
        m_bndlib = m_ggeo->getBndLib();
    }
    else
    {
        LOG(warning) << "GMaker::init booting from cache" ; 
        m_geolib = GGeoLib::load(m_cache);
        m_bndlib = GBndLib::load(m_cache, true );
    }
}


GSolid* GMaker::makeBox(glm::vec4& param)
{
    float size = param.w ; 
    gbbox bb(gfloat3(-size), gfloat3(size));  
    return makeBox(bb);
}

GSolid* GMaker::makeBox(gbbox& bbox)
{
    LOG(debug) << "GMaker::makeBox" ;

    unsigned int nvert = 24 ; 
    unsigned int nface = 6*2 ; 

    gfloat3* vertices = new gfloat3[nvert] ;
    guint3* faces = new guint3[nface] ;
    gfloat3* normals = new gfloat3[nvert] ;

    // TODO: migrate to NTrianglesNPY::cube ?
    GBBoxMesh::twentyfour(bbox, vertices, faces, normals );

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, nface,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  


    // TODO: tranform hookup with NTrianglesNPY 
    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}


GSolid* GMaker::makeSubdivSphere(glm::vec4& param, unsigned int nsubdiv, const char* type)
{
    LOG(info) << "GMaker::makeSphere" 
              << " nsubdiv " << nsubdiv
              << " type " << type
              << " param " << gformat(param) 
              ;

    NTrianglesNPY* tris = makeSubdivSphere(nsubdiv, type);

    float radius = param.w ; 
    float zpos   = param.z ;

    glm::vec3 scale(radius);
    glm::vec3 translate(0,0,zpos);
    tris->setTransform(scale, translate);   

    return makeSphere(tris);
}



NTrianglesNPY* GMaker::makeSubdivSphere(unsigned int nsubdiv, const char* type)
{
    NTrianglesNPY* tris(NULL);
    if(strcmp(type,"I")==0)
    {
        unsigned int ntri = 20*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* icos = NTrianglesNPY::icosahedron();
        tris = icos->subdivide(nsubdiv);  // (subdiv, ntri)  (0,20) (3,1280)
        assert(tris->getNumTriangles() == ntri);
    }
    else if(strcmp(type,"O")==0)
    {
        unsigned int ntri = 8*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* octa = NTrianglesNPY::octahedron();
        tris = octa->subdivide(nsubdiv); 
        assert(tris->getNumTriangles() == ntri);
    }
    else if(strcmp(type,"HO")==0)
    {
        unsigned int ntri = 4*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* ho = NTrianglesNPY::hemi_octahedron(); 
        tris = ho->subdivide(nsubdiv); 
        assert(tris->getNumTriangles() == ntri);
    }
    else if(strcmp(type,"HOS")==0)
    {
        unsigned int ntri = 4*(1 << (nsubdiv * 2)) ;
        glm::vec3 tr(0.,0.,0.5);
        glm::vec3 sc(1.,1.,1.);
        glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
        NTrianglesNPY* ho = NTrianglesNPY::hemi_octahedron(); 
        NTrianglesNPY* tho = ho->transform(m);
        tris = tho->subdivide(nsubdiv); 
        assert(tris->getNumTriangles() == ntri);
    }
    else if(strcmp(type,"C")==0)
    {
        unsigned int ntri = 2*6*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* cube = NTrianglesNPY::cube();
        tris = cube->subdivide(nsubdiv); 
        assert(tris->getNumTriangles() == ntri);
    }
    assert(tris);
    return tris ; 
}


GSolid* GMaker::makeZSphere(glm::vec4& param)
{
    NTrianglesNPY* ll = NTrianglesNPY::sphere(param);
    NTrianglesNPY* dk = NTrianglesNPY::disk(param);
    ll->add(dk);
    return makeSphere(ll);
}


void GMaker::makeZSphereIntersect(std::vector<GSolid*>& solids,  glm::vec4& param, const char* spec)
{
    float a_radius = param.x ; 
    float b_radius = param.y ; 
    float a_zpos   = param.z ;
    float b_zpos   = param.w ; 

    assert(b_zpos > a_zpos); 

    /*                           
                 +------------+   B
          A     /              \
           +---/---+            \
          /   /     \            |
         /   /       \           |
        |    |       |           | 
        |    |       |           |

    */

    nsphere a(0,0,a_zpos,a_radius);
    nsphere b(0,0,b_zpos,b_radius);
    ndisc d = nsphere::intersect(a,b) ;
    float zd = d.z();

    npart ar = a.zrhs(d);
    npart bl = b.zlhs(d);

    glm::vec4 arhs_param( a.costheta(zd), 1.f, a.z(), a.radius()) ;
    glm::vec4 blhs_param( -1, b.costheta(zd),  b.z(), b.radius()) ;

    NTrianglesNPY* a_tris = NTrianglesNPY::sphere(arhs_param);
    NTrianglesNPY* b_tris = NTrianglesNPY::sphere(blhs_param);

    GSolid* a_solid = makeSphere(a_tris);
    GSolid* b_solid = makeSphere(b_tris);

    float bbscale = 1.00001f ;  // TODO: currently ignored
    GParts* a_pts = GParts::make(ar, spec, bbscale);
    GParts* b_pts = GParts::make(bl, spec, bbscale);

    a_solid->setParts(a_pts);
    b_solid->setParts(b_pts);

    solids.push_back(a_solid);
    solids.push_back(b_solid);

}


GSolid* GMaker::makeSphere(NTrianglesNPY* tris)
{
    // TODO: generalize to makeSolid by finding other way to handle normals ?

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    NPY<float>* triangles = tris->getBuffer();

    glm::mat4 txf = tris->getTransform(); 

    GMesh* mesh = GMesh::make_spherelocal_mesh(triangles, meshindex);

    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    transform->Summary("GMaker::makeSphere");

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}


