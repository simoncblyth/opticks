#include <algorithm>
#include <iostream>

#include "PyMCubes/marchingcubes.hpp"

#include "NGLM.hpp"
#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"

#include "NSphere.hpp"


//template <typename T>
//NMarchingCubesNPY<T>::NMarchingCubesNPY()
NMarchingCubesNPY::NMarchingCubesNPY()
{
}




//template<typename T>
//NTrianglesNPY* NMarchingCubesNPY<T>::march(T sdf, const glm::uvec3& param, const glm::vec3& low, const glm::vec3& high )
NTrianglesNPY* NMarchingCubesNPY::march(SDFPtr sdf, const glm::uvec3& param, const glm::vec3& low, const glm::vec3& high )
{
    double lower_[3] ;
    double upper_[3] ;

    lower_[0] = low.x ; 
    lower_[1] = low.y ; 
    lower_[2] = low.z ; 

    upper_[0] = high.x ; 
    upper_[1] = high.y ; 
    upper_[2] = high.z ; 

    int numx = param.x ; 
    int numy = param.y ; 
    int numz = param.z ; 

    double isovalue = 0. ; 
    std::vector<double> vertices ; 
    std::vector<size_t> polygons ; 

    mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, sdf, isovalue, vertices, polygons);

    std::cout << " vertices " << vertices.size() << std::endl ; 
    std::cout << " polygons " << polygons.size() << std::endl ; 

    assert( polygons.size() % 3 == 0) ;

    unsigned ntri = polygons.size() / 3 ; 

    std::vector<size_t>::iterator  pmin = std::min_element(std::begin(polygons), std::end(polygons));
    std::vector<size_t>::iterator  pmax = std::max_element(std::begin(polygons), std::end(polygons));

    size_t imin = std::distance(std::begin(polygons), pmin) ;
    size_t imax = std::distance(std::begin(polygons), pmax) ;

    std::cout << "min element at: " << imin << " " << polygons[imin] << std::endl ; 
    std::cout << "max element at: " << imax << " " << polygons[imax] << std::endl ;

    NTrianglesNPY* tris = new NTrianglesNPY();

    std::vector<double>& v = vertices ; 

    for(unsigned t=0 ; t < ntri ; t++)
    {
         unsigned i0 = polygons[t*3 + 0];
         unsigned i1 = polygons[t*3 + 1];
         unsigned i2 = polygons[t*3 + 2];

         glm::vec3 v0( v[i0*3+0],  v[i0*3+1],  v[i0*3+2] );
         glm::vec3 v1( v[i1*3+0],  v[i1*3+1],  v[i1*3+2] );
         glm::vec3 v2( v[i2*3+0],  v[i2*3+1],  v[i2*3+2] );

         /*
         std::cout 
             << " t " << std::setw(5) << t 
             << " i0 " << std::setw(5) << i0  << " " << gformat(v0)
             << " i1 " << std::setw(5) << i1  << " " << gformat(v1)
             << " i2 " << std::setw(5) << i2  << " " << gformat(v2)
             << std::endl  ;
         */

         tris->add( v0, v1, v2 );
    }
    return tris ; 
}


/*
template class NMarchingCubesNPY<nsdf>;

template class NMarchingCubesNPY<nsphere>;
template class NMarchingCubesNPY<nunion>;
template class NMarchingCubesNPY<nintersection>;
template class NMarchingCubesNPY<ndifference>;
*/



