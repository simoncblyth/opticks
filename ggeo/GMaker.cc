
// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "NCSG.hpp"

#include "BParameters.hh"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPrism.hpp"
#include "NPart.hpp"

#include "OpticksCSG.h"

// ggeo-
#include "GBndLib.hh"
#include "GParts.hh"
#include "GBBoxMesh.hh"
#include "GMesh.hh"
#include "GMeshMaker.hh"
#include "GVolume.hh"
#include "GVector.hh"
#include "GMatrix.hh"
#include "GMaker.hh"

#include "PLOG.hh"


GMaker::GMaker(Opticks* ok, GBndLib* blib)
    :
    m_ok(ok),
    m_bndlib(blib)   // required for adding boundaries and returning indices for them 
{
    init();
}


void GMaker::init()
{
    assert( m_bndlib );
}



GVolume* GMaker::make(unsigned int /*index*/, OpticksCSG_t type, glm::vec4& param, const char* spec )
{
    // invoked from eg GGeoTest::createBoxInBox while looping over configured shape/boundary/param entries
    // for CSG triangulation need to be given the tree.. 
    //
    // NB note the the move to SDF means that forming a polygonization
    // no longer requires this, instead must implement the SDF : signed distance function

     GVolume* volume = NULL ; 
     switch(type)
     {
         case CSG_BOX:          volume = makeBox(param); break;
         case CSG_PRISM:        volume = makePrism(param, spec); break;
         case CSG_SPHERE:       volume = makeSubdivSphere(param, 3, "I") ; break; // I:icosahedron O:octahedron HO:hemi-octahedron C:cube 
         case CSG_ZSPHERE:      volume = makeZSphere(param) ; break;
         case CSG_ZLENS:        volume = makeZSphereIntersect_DEAD(param, spec) ; break;   // composite handled by adding child node
         case CSG_INTERSECTION: volume = makeBox(param); break ;    // boolean intersect
         case CSG_UNION:        volume = makeBox(param); break ;    // boolean union
         case CSG_DIFFERENCE:   volume = makeBox(param); break ;    // boolean difference

         case CSG_ZERO:         
         case CSG_PARTLIST:
         case CSG_PMT:
         case CSG_TUBS:
         case CSG_CYLINDER:
         case CSG_DISC:
         case CSG_SLAB:
         case CSG_CONE:
         case CSG_MULTICONE:
         case CSG_TRAPEZOID:
         case CSG_CONVEXPOLYHEDRON:
         case CSG_SEGMENT:
         case CSG_BOX3:
         case CSG_PLANE:
         case CSG_UNDEFINED:
         case CSG_FLAGPARTLIST:
         case CSG_FLAGNODETREE:
         case CSG_FLAGINVISIBLE:
         case CSG_TORUS:
         case CSG_ELLIPSOID:
         case CSG_HYPERBOLOID:
         case CSG_CUBIC:
                                volume = NULL ; break ;
     }
     assert(volume);
     volume->setCSGFlag( type );

     // TODO: most parts alread hooked up above, do this uniformly
     GParts* pts = volume->getParts();  
     if(pts == NULL)
     {
         pts = GParts::make(type, param, spec);  // (1,4,4) with typecode and bbox set 
         volume->setParts(pts);
     }
     assert(pts);

     unsigned boundary = m_bndlib->addBoundary(spec);  // only adds if not existing
     volume->setBoundaryAll(boundary);   // All loops over immediate children, needed for composite
     pts->setBoundaryAll(boundary);
     pts->enlargeBBoxAll(0.01f );

     return volume ; 
}


GVolume* GMaker::makeFromCSG(NCSG* csg, unsigned verbosity)
{
    return makeFromCSG(csg, m_bndlib, verbosity );
}


std::string GMaker::LVName(const char* shapename, int idx)
{
    return NCSG::TestVolumeName(shapename, "lv", idx );
}

std::string GMaker::PVName(const char* shapename, int idx)
{
    return NCSG::TestVolumeName(shapename, "pv", idx );
}

GVolume* GMaker::makeFromCSG(NCSG* csg, GBndLib* /*bndlib*/, unsigned verbosity )
{
    unsigned index = csg->getIndex();

    const char* spec = csg->getBoundary();  

    NTrianglesNPY* tris = csg->polygonize();

    GMesh* mesh = GMeshMaker::make_mesh(tris->getTris(), index);

    mesh->setCSG(csg);

    glm::mat4 txf = tris->getTransform(); 

    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    GVolume* volume = new GVolume(index, transform, mesh, UINT_MAX, NULL );     
    // csg is mesh-qty not a node-qty, boundary spec is a node-qty : so this is just for testing

    volume->setSensor( NULL );      

    // FORMERLY set boundary here 
    // moved later to allow relocation of basis
    // surfaces prior to arriving at a boundary index 


    OpticksCSG_t type = csg->getRootType() ;

    std::string lvn = csg->getTestLVName();
    std::string pvn = csg->getTestPVName();
    
    volume->setPVName( strdup(pvn.c_str()) );
    volume->setLVName( strdup(lvn.c_str()) );
    volume->setCSGFlag( type );

    GParts* pts = GParts::make( csg, spec, verbosity );

    volume->setParts( pts );


    LOG(info) << "GMaker::makeFromCSG" 
              << " verbosity " << verbosity 
              << " index " << index 
              << " boundary-spec " << spec 
              << " numTris " << ( tris ? tris->getNumTriangles() : 0 )
              << " trisMsg " << ( tris ? tris->getMessage() : "" )
              ; 

    return volume ; 
}





GVolume* GMaker::makeBox(glm::vec4& param)
{
    float size = param.w ; 

    gfloat3 mn( param.x - size, param.y - size, param.z - size );
    gfloat3 mx( param.x + size, param.y + size, param.z + size );

    //gbbox bb(gfloat3(-size), gfloat3(size));  
    gbbox bb(mn, mx);  

    return makeBox(bb);
}


GVolume* GMaker::makeBox(gbbox& bbox)
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

    GVolume* volume = new GVolume(nodeindex, transform, mesh, UINT_MAX, NULL );     

    volume->setBoundary(0);     // unlike ctor these create arrays
    volume->setSensor( NULL );      



    return volume ; 
}


GVolume* GMaker::makePrism(glm::vec4& param, const char* spec)
{
   /*
   */

    NTrianglesNPY* tris = NTrianglesNPY::prism(param);

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    GMesh* mesh = GMeshMaker::make_mesh(tris->getTris(), meshindex);
    //mesh->dumpNormals("GMaker::makePrism normals", 24);

    glm::mat4 txf = tris->getTransform(); 
    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    GVolume* volume = new GVolume(nodeindex, transform, mesh, UINT_MAX, NULL );     
    volume->setBoundary(0);     // unlike ctor these create arrays
    volume->setSensor( NULL );      

    nprism prism(param.x, param.y, param.z, param.w);
    npart  pprism = prism.part();

    GParts* pts = GParts::make(pprism, spec);

    volume->setParts(pts);

    return volume ; 
}







GVolume* GMaker::makeSubdivSphere(glm::vec4& param, unsigned int nsubdiv, const char* type)
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


GVolume* GMaker::makeZSphere(glm::vec4& param)
{
    NTrianglesNPY* ll = NTrianglesNPY::sphere(param);
    NTrianglesNPY* dk = NTrianglesNPY::disk(param);
    ll->add(dk);
    return makeSphere(ll);
}


void GMaker::makeBooleanComposite(char shapecode, std::vector<GVolume*>& /*volumes*/,  glm::vec4& /*param*/, const char* /*spec*/)
{
    assert( shapecode == 'I' || shapecode == 'J' || shapecode == 'K' );

    // hmm rustling up a trianglulated boolean composite is real difficult to do in general, 
    // but tis just for viz so could use bbox placeholder ?
}

GVolume* GMaker::makeZSphereIntersect_DEAD(glm::vec4& param, const char* spec)
{
    assert(0 && "NOT WORKING, NO POINT FIXING : AS NCSG APPROACH SO MUCH BETTER SEE tlens-- " );

    // parameters of two spheres with offset z positions used
    // to create a lens shape composed of 2 back-to-back zspheres

    float a_radius = param.x ; 
    float b_radius = param.y ; 
    float a_zpos   = param.z ;
    float b_zpos   = param.w ; 

    LOG(info) << "GMaker::makeZSphereIntersect"
              << " a_radius " << a_radius 
              << " b_radius " << b_radius 
              << " a_zpos " << a_zpos
              << " b_zpos " << b_zpos
              ;


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

    // TODO: such npy- level geo-specifics should
    //       be moved into npy- and handled at a higher
    //       from here

    nsphere* a = make_sphere(0,0,a_zpos,a_radius);
    nsphere* b = make_sphere(0,0,b_zpos,b_radius);
    ndisk* d = nsphere::intersect(a,b) ;   // from NPlane.hpp, not same as ndisc (degenerated ncylinder)
    float zd = d->z();

    d->dump("ndisk from nsphere::intersect" ); 
    LOG(info) << "ndisk::dump DONE " ; 


    // two CSG_SPHERE part
    npart ar = a->zrhs(d); 
    npart bl = b->zlhs(d);

    ar.dump("ar: a.zrhs(d) ");
    bl.dump("bl: b.zlhs(d) ");


    glm::vec3 a_center = a->center();
    glm::vec3 b_center = b->center();

    // ctmin, ctmax, zpos, radius
    glm::vec4 arhs_param( a->costheta(zd), 1.f, a_center.z, a->radius()) ;
    glm::vec4 blhs_param( -1, b->costheta(zd),  b_center.z, b->radius()) ;

    NTrianglesNPY* a_tris = NTrianglesNPY::sphere(arhs_param);
    NTrianglesNPY* b_tris = NTrianglesNPY::sphere(blhs_param);

    GVolume* a_volume = makeSphere(a_tris);
    GVolume* b_volume = makeSphere(b_tris);

    GParts* a_pts = GParts::make(ar, spec);
    GParts* b_pts = GParts::make(bl, spec);

    a_volume->setParts(a_pts);
    b_volume->setParts(b_pts);

    //volumes.push_back(a_volume);
    //volumes.push_back(b_volume);

    a_volume->addChild(b_volume);
    return a_volume ; 
}


GVolume* GMaker::makeSphere(NTrianglesNPY* tris)
{
    // TODO: generalize to makeSolid by finding other way to handle normals ?

    unsigned int meshindex = 0 ; 
    unsigned int nodeindex = 0 ; 

    NPY<float>* triangles = tris->getTris();

    glm::mat4 txf = tris->getTransform(); 

    GMesh* mesh = GMeshMaker::make_spherelocal_mesh(triangles, meshindex);

    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));

    //transform->Summary("GMaker::makeSphere");

    GVolume* volume = new GVolume(nodeindex, transform, mesh, UINT_MAX, NULL );     

    volume->setBoundary(0);     // unlike ctor these create arrays
    volume->setSensor( NULL );      

    return volume ; 
}

