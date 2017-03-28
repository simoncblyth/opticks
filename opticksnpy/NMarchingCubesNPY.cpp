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

NMarchingCubesNPY::NMarchingCubesNPY(int nx, int ny, int nz)
    :
    m_nx(nx),
    m_ny(ny > 0 ? ny : nx),
    m_nz(nz > 0 ? nz : nx),
    m_isovalue(0.),
    m_scale(1.01)      // without some space marching against a box gives zero triangles
{
}


NTrianglesNPY* NMarchingCubesNPY::operator()(nnode* node)
{
    nbbox bb = node->bbox();  // correctly gets the overloaded method

    m_lower[0] = bb.min.x*m_scale ; 
    m_lower[1] = bb.min.y*m_scale ; 
    m_lower[2] = bb.min.z*m_scale ; 

    m_upper[0] = bb.max.x*m_scale ; 
    m_upper[1] = bb.max.y*m_scale ; 
    m_upper[2] = bb.max.z*m_scale ; 

    LOG(info) << "NMarchingCubesNPY "
              << " bb " << bb.desc()
              << " scale " << m_scale 
              << " lower (" << m_lower[0] << "," << m_lower[1] << "," << m_lower[2] << ")"
              << " upper (" << m_upper[0] << "," << m_upper[1] << "," << m_upper[2] << ")"
              ;

    m_vertices.clear();
    m_polygons.clear();

    march(node);

    unsigned nvert = m_vertices.size() ;
    unsigned npoly = m_polygons.size() ;

    NTrianglesNPY* tris = NULL ; 

    if(nvert == 0 || npoly == 0)
    {
        LOG(warning) << "NMarchingCubesNPY gave zero verts/poly  "
                     << " nvert " << nvert
                     << " npoly " << npoly
                     << " MAKING PLACEHOLDER BBOX TRIS "  
                     ;
        tris = NTrianglesNPY::box(bb);
    } 
    else
    {
         tris = makeTriangles();
    }

    unsigned ntri = tris->getNumTriangles();
    nbbox* tris_bb = tris->findBBox(); 
    assert(tris_bb);

    LOG(info) << "NMarchingCubesNPY " 
              << " ntri " << std::setw(6) << ntri
              << " nvert " << std::setw(6) << nvert
              << " npoly " << std::setw(6) << npoly
              << " source node bb " << bb.desc() 
              << " output tris bb " << tris_bb->desc()
              ; 

    assert(ntri > 0);
    return tris ;
}

void NMarchingCubesNPY::march(nnode* node)
{
    // attempting to do the below without the upcasting gives segv 
    // suspect due to passing the functor by value into marching cubes ?
    switch(node->type)
    {
        case CSG_UNION:
            {
                nunion* n = (nunion*)node ; 
                mc::marching_cubes<double>(m_lower, m_upper, m_nx, m_ny, m_nz, *n, m_isovalue, m_vertices, m_polygons);
            }
            break ;
        case CSG_INTERSECTION:
            {
                nintersection* n = (nintersection*)node ; 
                mc::marching_cubes<double>(m_lower, m_upper, m_nx, m_ny, m_nz, *n, m_isovalue, m_vertices, m_polygons);
            }
            break ;
        case CSG_DIFFERENCE:
            {
                ndifference* n = (ndifference*)node ; 
                mc::marching_cubes<double>(m_lower, m_upper, m_nx, m_ny, m_nz, *n, m_isovalue, m_vertices, m_polygons);
            }
            break ;
        case CSG_SPHERE:
            {
                nsphere* n = (nsphere*)node ; 
                mc::marching_cubes<double>(m_lower, m_upper, m_nx, m_ny, m_nz, *n, m_isovalue, m_vertices, m_polygons);
            }
            break ;
        case CSG_BOX:
            {
                nbox* n = (nbox*)node ; 
                mc::marching_cubes<double>(m_lower, m_upper, m_nx, m_ny, m_nz, *n, m_isovalue, m_vertices, m_polygons);
            }
            break ;
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
}



NTrianglesNPY* NMarchingCubesNPY::makeTriangles()
{
    unsigned npol = m_polygons.size() ; 

    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    std::vector<size_t>::iterator  pmin = std::min_element(std::begin(m_polygons), std::end(m_polygons));
    std::vector<size_t>::iterator  pmax = std::max_element(std::begin(m_polygons), std::end(m_polygons));

    size_t imin = std::distance(std::begin(m_polygons), pmin) ;
    size_t imax = std::distance(std::begin(m_polygons), pmax) ;

    LOG(debug) << "min element at: " << imin << " " << m_polygons[imin] ; 
    LOG(debug) << "max element at: " << imax << " " << m_polygons[imax] ;

    NTrianglesNPY* tris = new NTrianglesNPY();

    std::vector<double>& v = m_vertices ; 

    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = m_polygons[t*3 + 0];
         unsigned i1 = m_polygons[t*3 + 1];
         unsigned i2 = m_polygons[t*3 + 2];

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


