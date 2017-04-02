
// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NCSG.hpp"

#include "NParameters.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPrism.hpp"
#include "NPart.hpp"

#include "NPolygonizer.hpp"


#include "OpticksCSG.h"

// ggeo-
#include "GGeo.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GParts.hh"
#include "GBBoxMesh.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GVector.hh"
#include "GMatrix.hh"
#include "GMaker.hh"

#include "PLOG.hh"

/*
bool GMaker::IsCompositeShape(char shapecode)
{
    return shapecode == 'L' ; 
}
bool GMaker::IsBooleanShape(char shapecode)
{
    return shapecode == 'I' || shapecode == 'J' || shapecode == 'K'  ;
}
*/


GSolid* GMaker::make(unsigned int index, char csgChar, glm::vec4& param, const char* spec )
{
    LOG(warning) << "GMaker::make OLD FORM IN USE csgChar " << csgChar ;  
    OpticksCSG_t type = CSGFlag(csgChar); 
    return GMaker::make( index, type, param, spec );
}



GSolid* GMaker::make(unsigned int /*index*/, OpticksCSG_t type, glm::vec4& param, const char* spec )
{
    // invoked from eg GGeoTest::createBoxInBox while looping over configured shape/boundary/param entries

    // for CSG triangulation need to be given the tree.. 

     GSolid* solid = NULL ; 
     switch(type)
     {
         case CSG_BOX:          solid = makeBox(param); break;
         case CSG_PRISM:        solid = makePrism(param, spec); break;
         case CSG_SPHERE:       solid = makeSubdivSphere(param, 3, "I") ; break; // I:icosahedron O:octahedron HO:hemi-octahedron C:cube 
         case CSG_ZSPHERE:      solid = makeZSphere(param) ; break;
         case CSG_ZLENS:        solid = makeZSphereIntersect(param, spec) ; break;   // composite handled by adding child node
         case CSG_INTERSECTION: solid = makeBox(param); break ;    // boolean intersect
         case CSG_UNION:        solid = makeBox(param); break ;    // boolean union
         case CSG_DIFFERENCE:   solid = makeBox(param); break ;    // boolean difference

         case CSG_ZERO:         
         case CSG_PARTLIST:
         case CSG_PMT:
         case CSG_TUBS:
         case CSG_UNDEFINED:
                                solid = NULL ; break ;
     }
     assert(solid);
     solid->setCSGFlag( type );

     // TODO: most parts alread hooked up above, do this uniformly
     GParts* pts = solid->getParts();  
     if(pts == NULL)
     {
         pts = GParts::make(type, param, spec);  // (1,4,4) with typecode and bbox set 
         solid->setParts(pts);
     }
     assert(pts);

     unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing
     solid->setBoundaryAll(boundary);   // All loops over immediate children, needed for composite
     pts->setBoundaryAll(boundary);
     pts->enlargeBBoxAll(0.01f );

     return solid ; 
}

GSolid* GMaker::makeFromCSG(NCSG* csg)
{
    unsigned index = csg->getIndex();

    LOG(info) << "GMaker::makeFromCSG" 
              << " index " << index 
              ; 

    NPolygonizer pg(csg);

    NTrianglesNPY* tris = pg.polygonize();

    nnode* root = csg->getRoot() ;

    GMesh* mesh = GMesh::make_mesh(tris->getTris(), index);


    glm::mat4 txf = tris->getTransform(); 
    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    GSolid* solid = new GSolid(index, transform, mesh, UINT_MAX, NULL );     


    const char* spec = csg->getBoundary();

    unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing

    solid->setBoundary(boundary);     // unlike ctor these create arrays

    solid->setSensor( NULL );      

    solid->setCSGFlag( root->type );
  
    GParts* pts = GParts::make( csg );

    solid->setParts( pts );

    // TODO: fix vagueness regards GMaker and GGeoTest responsibilities

    return solid ; 
}




GMaker::GMaker(Opticks* opticks, GGeo* ggeo)
    :
    m_opticks(opticks),
    m_ggeo(ggeo),
    m_geolib(NULL),
    m_bndlib(NULL)
{
    init();
}




void GMaker::init()
{
    if(m_ggeo)
    {
        LOG(warning) << "GMaker::init booting from m_ggeo" ; 
        m_geolib = m_ggeo->getGeoLib();
        m_bndlib = m_ggeo->getBndLib();
    }
    else
    {
        LOG(warning) << "GMaker::init booting from cache" ; 
        m_geolib = GGeoLib::load(m_opticks);
        m_bndlib = GBndLib::load(m_opticks, true );
    }
}


GSolid* GMaker::makeBox(glm::vec4& param)
{
    float size = param.w ; 

    gfloat3 mn( param.x - size, param.y - size, param.z - size );
    gfloat3 mx( param.x + size, param.y + size, param.z + size );

    //gbbox bb(gfloat3(-size), gfloat3(size));  
    gbbox bb(mn, mx);  

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


GSolid* GMaker::makePrism(glm::vec4& param, const char* spec)
{
   /*
   */

    NTrianglesNPY* tris = NTrianglesNPY::prism(param);

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    GMesh* mesh = GMesh::make_mesh(tris->getBuffer(), meshindex);
    //mesh->dumpNormals("GMaker::makePrism normals", 24);

    glm::mat4 txf = tris->getTransform(); 
    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     
    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    nprism prism(param.x, param.y, param.z, param.w);
    npart  pprism = prism.part();

    GParts* pts = GParts::make(pprism, spec);

    solid->setParts(pts);

    return solid ; 
}







GSolid* GMaker::makeSubdivSphere(glm::vec4& param, unsigned int nsubdiv, const char* type)
{
    LOG(debug) << "GMaker::makeSubdivSphere" 
              << " nsubdiv " << nsubdiv
              << " type " << type
              << " param " << gformat(param) 
              ;

    NTrianglesNPY* tris = makeSubdivSphere(nsubdiv, type);

    float radius = param.w ; 

    glm::vec3 scale(radius);
    glm::vec3 translate(param.x,param.y,param.z);  // formerly only z shifts were honoured
    tris->setTransform(scale, translate);   

    return makeSphere(tris);
}



NTrianglesNPY* GMaker::makeSubdivSphere(unsigned int nsubdiv, const char* type)
{
    // approach to using geodesic subdiv for partial spheres 
    // http://www.unitbv.ro/faculties/biblio/Buletin_univ/pdf/Iacob.pdf

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


void GMaker::makeBooleanComposite(char shapecode, std::vector<GSolid*>& /*solids*/,  glm::vec4& /*param*/, const char* /*spec*/)
{
    assert( shapecode == 'I' || shapecode == 'J' || shapecode == 'K' );

    // hmm rustling up a trianglulated boolean composite is real difficult to do in general, 
    // but tis just for viz so could use bbox placeholder ?
}

GSolid* GMaker::makeZSphereIntersect(glm::vec4& param, const char* spec)
{
    // parameters of two spheres with offset z positions used
    // to create

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

    nsphere a = make_nsphere(0,0,a_zpos,a_radius);
    nsphere b = make_nsphere(0,0,b_zpos,b_radius);
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

    GParts* a_pts = GParts::make(ar, spec);
    GParts* b_pts = GParts::make(bl, spec);

    a_solid->setParts(a_pts);
    b_solid->setParts(b_pts);

    //solids.push_back(a_solid);
    //solids.push_back(b_solid);

    a_solid->addChild(b_solid);
    return a_solid ; 
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

    //transform->Summary("GMaker::makeSphere");

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}









