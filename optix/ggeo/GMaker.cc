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
#include "NLog.hpp"

// npy-
#include "NPY.hpp"
#include "NSphere.hpp"
#include "NLog.hpp"

const char* GMaker::SPHERE = "sphere" ; 
const char* GMaker::BOX = "box" ; 
const char* GMaker::PMT = "pmt" ; 
const char* GMaker::UNDEFINED = "undefined" ; 
 
const char* GMaker::ShapeName(char shapecode)
{
    switch(shapecode) 
    {
       case 'B':return BOX    ; break ; 
       case 'S':return SPHERE ; break ; 
       case 'P':return PMT    ; break ; 
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
                  unsigned int nsubdiv = 3 ; 
                  char type = 'C' ;  // I:icosahedron O:octagon C:cube L:latlon (ignores nsubdiv)
                  solid = makeSphere(param, nsubdiv, type) ;
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



GSolid* GMaker::makeSphere(glm::vec4& param, unsigned int subdiv, char type)
{
    LOG(debug) << "GMaker::makeSphere" ;


    NPY<float>* triangles(NULL);

    if(type == 'I')
    {
        unsigned int ntri = 20*(1 << (subdiv * 2)) ;
        triangles = NSphere::icosahedron(subdiv);  // (subdiv, ntri)  (0,20) (3,1280)
        assert(triangles->getNumItems() == ntri);
    }
    else if(type == 'O')
    {
        unsigned int ntri = 8*(1 << (subdiv * 2)) ;
        triangles = NSphere::octahedron(subdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(type == 'C')
    {
        unsigned int ntri = 2*6*(1 << (subdiv * 2)) ;
        triangles = NSphere::cube(subdiv); 
        assert(triangles->getNumItems() == ntri);
    }
    else if(type == 'L')
    {
        unsigned int n_polar = 24 ; 
        unsigned int n_azimuthal = 24 ; 
        unsigned int ntri = n_polar*(n_azimuthal-1)*2 ; 
        triangles = NSphere::latlon(n_polar, n_azimuthal); 
        assert(triangles->getNumItems() == ntri);
    }

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





