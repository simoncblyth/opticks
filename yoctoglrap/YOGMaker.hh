#pragma once

#include <vector>

#include "YGLTF.h"
#include "YOG_API_EXPORT.hh"

struct YOGGeometry ;

namespace ygltf 
{
    struct glTF_t ;   
    struct fl_gltf ;
    struct fl_mesh ;
    struct node_t ;
    struct mesh_t ;
}

#include <vector>

/**

Did not pursue this direct to ygltf approach, 
trying instead a direct translation of sc.py over in YOG.hh
which separates (a little) the tree creation from the 
gltf creation.

**/

class YOG_API YOGMaker 
{
   public:
       static std::unique_ptr<ygltf::glTF_t> make_gltf(const YOGGeometry& geom );
   public:
       YOGMaker();
   private:
        ygltf::glTF_t*                    m_gltf ; 
        std::vector<ygltf::scene_t>&      m_scenes ;
        std::vector<ygltf::node_t>&       m_nodes ; 
        std::vector<ygltf::buffer_t>&     m_buffers ; 
        std::vector<ygltf::bufferView_t>& m_bufferViews ; 
        std::vector<ygltf::accessor_t>&   m_accessors ; 
        std::vector<ygltf::mesh_t>&       m_meshes ; 
        std::vector<ygltf::material_t>&   m_materials ; 
};





