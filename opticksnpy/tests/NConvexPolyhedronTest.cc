#include "NCSG.hpp"
#include "NConvexPolyhedron.hpp"
#include "NBBox.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NSceneConfig.hpp"

#include <iostream>
#include <iomanip>

#include "PLOG.hh"
#include "NPY_LOG.hh"



nconvexpolyhedron* test_load(const char* path)
{
    LOG(info) << "test_load " << path ;  

    const char* gltfconfig = "csg_bbox_parsurf=1" ;
    const NSceneConfig* config = new NSceneConfig(gltfconfig) ; 

    nnode* root = nnode::load(path, config );

    assert(root->type == CSG_CONVEXPOLYHEDRON );
    nconvexpolyhedron* cpol = (nconvexpolyhedron*)root  ;
    cpol->verbosity = 1 ;  
    cpol->pdump("cpol->pdump");

    return cpol ; 
}

nconvexpolyhedron test_make()
{
    nquad param ; 
    nquad param1 ; 
    nquad param2 ; 
    nquad param3 ; 

    param.u = {0,0,0,0} ;
    param1.u = {0,0,0,0} ;
    param2.u = {0,0,0,0} ;
    param3.u = {0,0,0,0} ;

    nconvexpolyhedron cpol = make_convexpolyhedron(param, param1, param2, param3 );
    return cpol ; 
}


void test_sdf(const nconvexpolyhedron* cpol)
{
    std::cout << "cpol(50,50,50) = " << (*cpol)(50,50,50) << std::endl ;
    std::cout << "cpol(-50,-50,-50) = " << (*cpol)(-50,-50,-50) << std::endl ;

    for(float v=-400.f ; v <= 400.f ; v+= 100.f )
    {
        float sd = (*cpol)(v,0,0) ;

        std::cout 
            << " x  " << std::setw(10) << v
            << " sd:  " << std::setw(10) << sd
/*
            << " y  " << std::setw(10) << (*cpol)(0,v,0)
            << " z  " << std::setw(10) << (*cpol)(0,0,v)
            << " xy " << std::setw(10) << (*cpol)(v,v,0)
            << " xz " << std::setw(10) << (*cpol)(v,0,v)
            << " yz " << std::setw(10) << (*cpol)(0,v,v)
            << "xyz " << std::setw(10) << (*cpol)(v,v,v)
*/
            << std::endl ; 
    } 
}



void test_intersect(const nconvexpolyhedron* cpol)
{
    glm::vec3 ray_origin(0,0,0);
    float t_min = 0.f ; 

    for(unsigned i=0 ; i < 3 ; i++)  
    {
        for(unsigned j=0 ; j < 2 ; j++)
        {
            glm::vec3 ray_direction(0,0,0);
            ray_direction.x = i == 0  ? (j == 0 ? 1 : -1 ) : 0 ; 
            ray_direction.y = i == 1  ? (j == 0 ? 1 : -1 ) : 0 ; 
            ray_direction.z = i == 2  ? (j == 0 ? 1 : -1 ) : 0 ; 
            std::cout << " dir " << ray_direction << std::endl ; 

            glm::vec4 isect(0.f);
            bool valid_intersect = cpol->intersect(  t_min , ray_origin, ray_direction , isect );
            assert(valid_intersect);

            std::cout << " isect : " << isect << std::endl ; 
        }
    }
}


void test_bbox(const nconvexpolyhedron* cpol)
{
    nbbox bb = cpol->bbox() ; 
    std::cout << bb.desc() << std::endl ; 
}


void test_getSurfacePointsAll(nconvexpolyhedron* cpol)
{
    cpol->dump_planes(); 


    unsigned level = 1 ;  // +---+---+
    int margin = 1 ;      // o---*---o
    unsigned verbosity = 1 ; 
    unsigned prim_idx = 0 ; 

    cpol->collectParPoints(prim_idx, level, margin, FRAME_LOCAL, verbosity);
    const std::vector<glm::vec3>& surf = cpol->par_points ; 

    LOG(info) << "test_parametric"
              << " surf points " << surf.size()
              ;

    for(unsigned i=0 ; i < surf.size() ; i++ ) std::cout << gpresent(surf[i]) << std::endl ; 

}


void test_dumpSurfacePointsAll(const nconvexpolyhedron* cpol)
{
    cpol->dumpSurfacePointsAll("dumpSurfacePointsAll", FRAME_LOCAL);
}




nconvexpolyhedron* test_make_trapezoid()
{
   /*
    z-order verts


                  6----------7
                 /|         /|
                / |        / |
               4----------5  |
               |  |       |  |                       
               |  |       |  |         Z    
               |  2-------|--3         |  Y
               | /        | /          | /
               |/         |/           |/
               0----------1            +------ X
                         

    x1: x length at -z
    y1: y length at -z

    x2: x length at +z
    y2: y length at +z

    z:  z length

    */

    float z  = 100 ; 
    float x1 = 200 ; 
    float y1 = 200 ; 
    float x2 = 200 ; 
    float y2 = 200 ; 
  
    nconvexpolyhedron* cpol = nconvexpolyhedron::make_trapezoid( z,  x1,  y1,  x2,  y2 );

    cpol->dump_planes();

    return cpol ; 
}


void test_transform_planes(nconvexpolyhedron* cpol )
{
    NPY<float>* planbuf = NPY<float>::make(cpol->planes);
    planbuf->dump("before");

    glm::mat4 placement = nglmext::make_translate(1002,-5000,10);

    nglmext::transform_planes(planbuf, placement );

    planbuf->dump("after");
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //const char* path = argc > 1 ?  argv[1] : "$TMP/tboolean-trapezoid--/1" ;
    // nconvexpolyhedron*  cpol = test_load(path)
    //
    //test_sdf(cpol);
    //test_intersect(cpol);
    //test_bbox(cpol);
    //test_getSurfacePointsAll(cpol);
    //test_dumpSurfacePointsAll(cpol);

    nconvexpolyhedron* cpol = test_make_trapezoid();
    test_dumpSurfacePointsAll(cpol);

    //test_transform_planes(cpol);


    return 0 ; 
}

