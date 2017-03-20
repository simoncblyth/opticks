#include <algorithm>
#include <iostream>

#include "PyMCubes/marchingcubes.hpp"

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NMarchingCubesNPY.hpp"
#include "NTrianglesNPY.hpp"

#include "NNode.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"

#include "PLOG.hh"


NMarchingCubesNPY::NMarchingCubesNPY(const nuvec3& param) : m_param(param) {}


template<typename T>
NTrianglesNPY* NMarchingCubesNPY::operator()(T* node)
{
    double lower_[3] ;
    double upper_[3] ;

    nbbox bb = node->bbox();  // correctly gets the overloaded method

    double scale = 1.01 ;     // without some space marching against a box gives zero triangles

    lower_[0] = bb.min.x*scale ; 
    lower_[1] = bb.min.y*scale ; 
    lower_[2] = bb.min.z*scale ; 

    upper_[0] = bb.max.x*scale ; 
    upper_[1] = bb.max.y*scale ; 
    upper_[2] = bb.max.z*scale ; 

    int numx = m_param.x ; 
    int numy = m_param.y ; 
    int numz = m_param.z ; 

    double isovalue = 0. ; 
    std::vector<double> vertices ; 
    std::vector<size_t> polygons ; 

    // attempting to do the below without the upcasting gives segv 
    // suspect due to passing the functor by value into marching cubes ?
    switch(node->type)
    {
        case CSG_UNION:
            {
                nunion* n = (nunion*)node ; 
                mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, *n, isovalue, vertices, polygons);
            }
            break ;
        case CSG_INTERSECTION:
            {
                nintersection* n = (nintersection*)node ; 
                mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, *n, isovalue, vertices, polygons);
            }
            break ;
        case CSG_DIFFERENCE:
            {
                ndifference* n = (ndifference*)node ; 
                mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, *n, isovalue, vertices, polygons);
            }
            break ;
        case CSG_SPHERE:
            {
                nsphere* n = (nsphere*)node ; 
                mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, *n, isovalue, vertices, polygons);
            }
            break ;
        case CSG_BOX:
            {
                nbox* n = (nbox*)node ; 
                mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, *n, isovalue, vertices, polygons);
            }
            break ;
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }

    

    LOG(trace) << " vertices " << vertices.size() ; 
    LOG(trace) << " polygons " << polygons.size() ; 


    unsigned npol = polygons.size() ; 

    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    std::vector<size_t>::iterator  pmin = std::min_element(std::begin(polygons), std::end(polygons));
    std::vector<size_t>::iterator  pmax = std::max_element(std::begin(polygons), std::end(polygons));

    size_t imin = std::distance(std::begin(polygons), pmin) ;
    size_t imax = std::distance(std::begin(polygons), pmax) ;

    LOG(debug) << "min element at: " << imin << " " << polygons[imin] ; 
    LOG(debug) << "max element at: " << imax << " " << polygons[imax] ;

    NTrianglesNPY* tris = new NTrianglesNPY();

    std::vector<double>& v = vertices ; 

    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = polygons[t*3 + 0];
         unsigned i1 = polygons[t*3 + 1];
         unsigned i2 = polygons[t*3 + 2];

         glm::vec3 v0( v[i0*3+0],  v[i0*3+1],  v[i0*3+2] );
         glm::vec3 v1( v[i1*3+0],  v[i1*3+1],  v[i1*3+2] );
         glm::vec3 v2( v[i2*3+0],  v[i2*3+1],  v[i2*3+2] );

         LOG(trace)
             << " t " << std::setw(5) << t 
             << " i0 " << std::setw(5) << i0  << " " << gformat(v0)
             << " i1 " << std::setw(5) << i1  << " " << gformat(v1)
             << " i2 " << std::setw(5) << i2  << " " << gformat(v2)
             ;

         tris->add( v0, v1, v2 );
    }
    return tris ; 
}


template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<nnode>(nnode*);
template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<nsphere>(nsphere*);
template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<nbox>(nbox*);
template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<nunion>(nunion*);
template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<nintersection>(nintersection*);
template NPY_API NTrianglesNPY* NMarchingCubesNPY::operator()<ndifference>(ndifference*);



