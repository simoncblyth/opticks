#include <glm/glm.hpp>

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
#include "NSphere.hpp"
#include "NLog.hpp"


#include <glm/glm.hpp>



const char* GMaker::SPHERE = "sphere" ; 
const char* GMaker::ZSPHERE = "zsphere" ; 
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
       case 'P':return PMT     ; break ; 
       case 'U':return UNDEFINED ; break ;
    }
    return NULL ;
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


GSolid* GMaker::make(unsigned int index, char shapecode, glm::vec4& param, const char* spec )
{
    GSolid* solid = NULL ; 
    switch(shapecode)
    {
        case 'B': 
                  solid = makeBox(param);
                  break;
        case 'S': 
                  solid = makeSphere(param, 3, "O") ;
                  // I:icosahedron O:octahedron HO:hemi-octahedron C:cube L:latlon (ignores nsubdiv)
                  break;
        case 'Z':
                  solid = makeZSphere(param) ;
                  break;
    }
    assert(solid);
    unsigned int boundary = m_bndlib->addBoundary(spec);
    solid->setBoundary(boundary);
    solid->setIndex(index);

    float bbscale = 1.00001f ; 
    GParts* pts = GParts::make(shapecode, param, spec, bbscale);

    // hmm: single part per solid assumption here
    pts->setIndex(0u, solid->getIndex());
    pts->setNodeIndex(0u, solid->getIndex());
    pts->setBndLib(m_bndlib);

    solid->setParts(pts);

    return solid ; 
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

    GBBoxMesh::twentyfour(bbox, vertices, faces, normals );

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    GMesh* mesh = new GMesh(meshindex, vertices, nvert,  
                                       faces, nface,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[nvert]);
    mesh->setColor(0.5,0.5,0.5);  


    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}



GSolid* GMaker::makeSphere(glm::vec4& param, unsigned int subdiv, const char* type)
{
    LOG(debug) << "GMaker::makeSphere" ;


    NPY<float>* triangles(NULL);

    if(strcmp(type,"I")==0)
    {
        unsigned int ntri = 20*(1 << (subdiv * 2)) ;
        triangles = NSphere::icosahedron(subdiv);  // (subdiv, ntri)  (0,20) (3,1280)
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"O")==0)
    {
        unsigned int ntri = 8*(1 << (subdiv * 2)) ;
        triangles = NSphere::octahedron(subdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"HO")==0)
    {
        unsigned int ntri = 4*(1 << (subdiv * 2)) ;
        triangles = NSphere::hemi_octahedron(subdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"HOS")==0)
    {
        unsigned int ntri = 4*(1 << (subdiv * 2)) ;

        glm::vec3 tr(0.,0.,0.5);
        glm::vec3 sc(1.,1.,1.);

        // scale and then translate   (translation not scaled)
        glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

        // aim for spherical segments via subdiv, thwarted by curved edges
        // unclear what basis shape will do what want 

        triangles = NSphere::hemi_octahedron(subdiv, m); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"C")==0)
    {
        unsigned int ntri = 2*6*(1 << (subdiv * 2)) ;
        triangles = NSphere::cube(subdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"L")==0)
    {
        unsigned int n_polar = 24 ; 
        unsigned int n_azimuthal = 24 ; 
        unsigned int ntri = n_polar*(n_azimuthal-1)*2 ; 
        triangles = NSphere::latlon(n_polar, n_azimuthal); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"LZ")==0)
    {
        unsigned int n_polar = 24 ; 
        unsigned int n_azimuthal = 24 ; 
        triangles = NSphere::latlon(param.x, param.y, n_polar, n_azimuthal); 
    }

    assert(triangles);
    return makeSphere(param, triangles);
}



GSolid* GMaker::makeZSphere(glm::vec4& param)
{
    unsigned int n_polar = 24 ; 
    unsigned int n_azimuthal = 24 ; 

    NPY<float>* triangles = NSphere::latlon(param.x, param.y, n_polar, n_azimuthal); 
 
    return makeSphere(param, triangles);
}


GSolid* GMaker::makeSphere(glm::vec4& param, NPY<float>* triangles)
{
    float radius = param.w ; 

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    GMesh* mesh = GMesh::make_mesh(triangles, radius, meshindex);

    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}


