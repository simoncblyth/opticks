#include "NDualContouringSample.hpp"
#include "NTrianglesNPY.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "DualContouringSample/mesh.h"
#include "DualContouringSample/octree.h"

#include "NSphere.hpp"
#include "NNode.hpp"
#include "NBox.hpp"

#include "PLOG.hh"

NDualContouringSample::NDualContouringSample(unsigned log2size, float threshold)
  :
   m_octreeSize(1 << log2size),
   m_threshold(threshold)
{
}


NTrianglesNPY* NDualContouringSample::operator()(nnode* node)
{
    VertexBuffer vertices;
    IndexBuffer indices;

    vertices.clear();
    indices.clear();


    OctreeNode* root = NULL ; 

    switch(node->type)
    {
        case CSG_UNION:
            {
                nunion* n = (nunion*)node ; 
                std::function<float(float,float,float)> f = *n ; 
                root = BuildOctree(glm::ivec3(-m_octreeSize / 2), m_octreeSize, m_threshold, f ) ;
            }
            break ;
        case CSG_INTERSECTION:
            {
                nintersection* n = (nintersection*)node ; 
                std::function<float(float,float,float)> f = *n ; 
                root = BuildOctree(glm::ivec3(-m_octreeSize / 2), m_octreeSize, m_threshold, f ) ;
            }
            break ;
        case CSG_DIFFERENCE:
            {
                ndifference* n = (ndifference*)node ; 
                std::function<float(float,float,float)> f = *n ; 
                root = BuildOctree(glm::ivec3(-m_octreeSize / 2), m_octreeSize, m_threshold, f ) ;
            }
            break ;
        case CSG_SPHERE:
            {
                nsphere* n = (nsphere*)node ; 
                std::function<float(float,float,float)> f = *n ; 
                root = BuildOctree(glm::ivec3(-m_octreeSize / 2), m_octreeSize, m_threshold, f ) ;
            }
            break ;
        case CSG_BOX:
            {
                nbox* n = (nbox*)node ;  
                std::function<float(float,float,float)> f = *n ; 
                root = BuildOctree(glm::ivec3(-m_octreeSize / 2), m_octreeSize, m_threshold, f ) ;
            }
            break ;
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }

    assert(root);


    GenerateMeshFromOctree(root, vertices, indices);


    LOG(info) << " vertices " << vertices.size() ;
    LOG(info) << " indices  " << indices.size() ;


    unsigned npol = indices.size() ; 

    assert( npol % 3 == 0) ;
    unsigned ntri = npol / 3 ; 

    IndexBuffer::iterator  pmin = std::min_element(std::begin(indices), std::end(indices));
    IndexBuffer::iterator  pmax = std::max_element(std::begin(indices), std::end(indices));

    size_t imin = std::distance(std::begin(indices), pmin) ;
    size_t imax = std::distance(std::begin(indices), pmax) ;

    LOG(debug) << "min element at: " << imin << " " << indices[imin] ; 
    LOG(debug) << "max element at: " << imax << " " << indices[imax] ;


    NTrianglesNPY* tris = new NTrianglesNPY();

    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = indices[t*3 + 0];
         unsigned i1 = indices[t*3 + 1];
         unsigned i2 = indices[t*3 + 2];
          
         MeshVertex& v0 = vertices[i0] ;
         MeshVertex& v1 = vertices[i1] ;
         MeshVertex& v2 = vertices[i2] ;

         LOG(info)
             << " t " << std::setw(5) << t 
             << " i0 " << std::setw(5) << i0  << " " << gformat(v0.xyz)  << " " << gformat(v0.normal) 
             << " i1 " << std::setw(5) << i1  << " " << gformat(v1.xyz)  << " " << gformat(v1.normal) 
             << " i2 " << std::setw(5) << i2  << " " << gformat(v2.xyz)  << " " << gformat(v2.normal) 
             ;

         tris->add( v0.xyz, v1.xyz, v2.xyz );
         tris->addNormal( v0.normal, v1.normal, v2.normal );
    }
    return tris ; 
}


