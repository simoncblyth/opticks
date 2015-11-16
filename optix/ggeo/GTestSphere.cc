#include "GTestSphere.hh"

#include "GMesh.hh"
#include "GSolid.hh"
#include "GVector.hh"
#include "GMatrix.hh"

// npy-
#include "NPY.hpp"
#include "NSphere.hpp"
#include "NLog.hpp"

GSolid* GTestSphere::makeSolid(glm::vec4& spec, unsigned int meshindex, unsigned int nodeindex)
{
    LOG(debug) << "GTestSphere::makeSolid" ;

    unsigned int subdiv = 0 ; 
    unsigned int ntri = 20*(1 << (subdiv * 2)) ;
    NPY<float>* triangles = NSphere::icosahedron(0);  // (subdiv, ntri)  (0,20)
    assert(triangles->getNumItems() == ntri);

    float radius = spec.w ; 

    GMesh* mesh = GMesh::make_mesh(triangles, radius, meshindex);

    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, mesh, UINT_MAX, NULL );     

    solid->setBoundary(0);     // unlike ctor these create arrays
    solid->setSensor( NULL );      

    return solid ; 
}






