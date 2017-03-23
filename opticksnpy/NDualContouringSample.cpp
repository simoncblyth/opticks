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

NDualContouringSample::NDualContouringSample(unsigned log2size, float threshold, float scale_bb)
  :
   m_octreeSize(1 << log2size),
   m_threshold(threshold),
   m_scale_bb(scale_bb)
{
}







NTrianglesNPY* NDualContouringSample::operator()(nnode* node)
{
    m_node_bb = node->bbox();  // overloaded method 

    nvec4     bbce = m_node_bb.center_extent();
    float xyzExtent = bbce.w*m_scale_bb ;   // slightly enlarge, for cubes


    glm::ivec3 ilow(-m_octreeSize / 2);

    float ijkExtent = fabs(ilow.x) ;
    float ijk2xyz = xyzExtent/ijkExtent ;   // octree -> real world coordinates

    glm::vec4 ce(bbce.x, bbce.y, bbce.z, ijk2xyz );


    LOG(info) << "NDualContouringSample "
              << " xyzExtent " << xyzExtent
              << " ijkExtent " << ijkExtent
              << " bbce " << bbce.desc()
              << " ce " << gformat(ce)
              << " ilow " << gformat(ilow)
              ;
              

    VertexBuffer vertices;
    IndexBuffer indices;

    vertices.clear();
    indices.clear();

    std::function<float(float,float,float)> f = node->sdf();

    OctreeNode* octree = BuildOctree(ilow, m_octreeSize, m_threshold, f, ce ) ;


    NTrianglesNPY* tris = NULL ; 

    if(octree == NULL)
    {   
        LOG(warning) << "NDualContouringSample : NULL octree  "
                     << " for node " << CSGName(node->type)
                     << " MAKING PLACEHOLDER BBOX TRIS "  
                     ;
        tris = NTrianglesNPY::box(m_node_bb);
        return tris ; 
    }

    GenerateMeshFromOctree(octree, vertices, indices, ce);


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


    tris = new NTrianglesNPY();

    for(unsigned t=0 ; t < ntri ; t++)
    {
         assert( t*3+2 < npol );

         unsigned i0 = indices[t*3 + 0];
         unsigned i1 = indices[t*3 + 1];
         unsigned i2 = indices[t*3 + 2];
          
         MeshVertex& v0 = vertices[i0] ;
         MeshVertex& v1 = vertices[i1] ;
         MeshVertex& v2 = vertices[i2] ;

        /*
         LOG(info)
             << " t " << std::setw(5) << t 
             << " i0 " << std::setw(5) << i0  << " " << gformat(v0.xyz)  << " " << gformat(v0.normal) 
             << " i1 " << std::setw(5) << i1  << " " << gformat(v1.xyz)  << " " << gformat(v1.normal) 
             << " i2 " << std::setw(5) << i2  << " " << gformat(v2.xyz)  << " " << gformat(v2.normal) 
             ;
         */

         tris->add( v0.xyz, v1.xyz, v2.xyz );
         tris->addNormal( v0.normal, v1.normal, v2.normal );
    }
    return tris ; 
}


