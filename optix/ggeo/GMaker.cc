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
#include "NTrianglesNPY.hpp"
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


    solid->setParts(pts);  // hang the analytic on the triangulated
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



GSolid* GMaker::makeSphere(glm::vec4& param, unsigned int nsubdiv, const char* type)
{
    LOG(debug) << "GMaker::makeSphere" ;


    NPY<float>* triangles(NULL);

    if(strcmp(type,"I")==0)
    {
        unsigned int ntri = 20*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* icos = NTrianglesNPY::icosahedron();
        triangles = icos->subdivide(nsubdiv);  // (subdiv, ntri)  (0,20) (3,1280)
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"O")==0)
    {
        unsigned int ntri = 8*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* octa = NTrianglesNPY::octahedron();
        triangles = octa->subdivide(nsubdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"HO")==0)
    {
        unsigned int ntri = 4*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* ho = NTrianglesNPY::hemi_octahedron(); 
        triangles = ho->subdivide(nsubdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"HOS")==0)
    {
        unsigned int ntri = 4*(1 << (nsubdiv * 2)) ;

        glm::vec3 tr(0.,0.,0.5);
        glm::vec3 sc(1.,1.,1.);

        // scale and then translate   (translation not scaled)
        glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

        // aim for spherical segments via subdiv, thwarted by curved edges
        // unclear what basis shape will do what want 

        NTrianglesNPY* ho = NTrianglesNPY::hemi_octahedron(); 
        NTrianglesNPY* tho = ho->transform(m);
        triangles = tho->subdivide(nsubdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"C")==0)
    {
        unsigned int ntri = 2*6*(1 << (nsubdiv * 2)) ;
        NTrianglesNPY* cube = NTrianglesNPY::cube();
        triangles = cube->subdivide(nsubdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(strcmp(type,"L")==0)
    {
        NTrianglesNPY* ll = NTrianglesNPY::sphere();
        triangles = ll->getBuffer();
    }
    else if(strcmp(type,"LZ")==0)
    {
        NTrianglesNPY* ll = NTrianglesNPY::sphere(param);
        triangles = ll->getBuffer();
    }

    assert(triangles);

    NPY<float>* ttris = triangles->scale(param.w) ; // above deal in unit sphers

    return makeSphere(ttris);
}



GSolid* GMaker::makeZSphere(glm::vec4& param)
{
    NTrianglesNPY* ll = NTrianglesNPY::sphere(param);
    NTrianglesNPY* dk = NTrianglesNPY::disk(param);
    ll->add(dk);

    NPY<float>* tris = ll->getBuffer();
    NPY<float>* ttris = tris->scale(param.w) ; // radius

    return makeSphere(ttris);
}

GSolid* GMaker::makeSphere(NPY<float>* triangles)
{
    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    // TODO: this is assuming unit sphere for normals, fix by normalizing 
    GMesh* mesh = GMesh::make_mesh(triangles, 1.0, meshindex);

    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}


GSolid* GMaker::makeZSphereIntersect(glm::vec4& aparam, glm::vec4& bparam)
{
    NTrianglesNPY* a = NTrianglesNPY::sphere(aparam);
    NTrianglesNPY* b = NTrianglesNPY::sphere(bparam);

    NTrianglesNPY* tris = new NTrianglesNPY();

    // TODO: radius scaling with the add , translate too ? 
    tris->add(a);
    tris->add(b);


    NPY<float>* buf = tris->getBuffer(); 

    return makeSphere(buf);
}






