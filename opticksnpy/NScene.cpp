
#include "YoctoGL/yocto_gltf.h"

#include "BFile.hh"
#include "NScene.hpp"

#include "PLOG.hh"


NScene* NScene::load(const char* base, const char* name)
{
    std::string path = BFile::FormPath(base, name);

    LOG(info) << "NScene::load"
              << " path " << path
              ;

    bool load_bin = true ; 
    bool load_shaders = true ; 
    bool load_img = false ; 
    bool skip_missing = true  ;   

    ygltf::glTF_t* gltf = ygltf::load_gltf(path, load_bin, load_shaders, load_img, skip_missing ) ;

    return new NScene(gltf); 

}

NScene::NScene(ygltf::glTF_t* gltf )
   :
    m_gltf(gltf)
{
    walk();
}



/*
Using extracts from
/usr/local/env/graphics/yoctogl/yocto-gl/yocto/yocto_gltf.cpp 
*/

static inline std::array<float, 16> _float4x4_mul(
    const std::array<float, 16>& a, const std::array<float, 16>& b) {
    auto c = std::array<float, 16>();
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            c[j * 4 + i] = 0;
            for (auto k = 0; k < 4; k++)
                c[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
        }
    }
    return c;
}

const std::array<float, 16> _identity_float4x4 = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};



void NScene::walk()
{
    LOG(info) << "NScene::walk" ; 

    int scn_id = 0 ; 

    auto scn = &m_gltf->scenes.at(scn_id);

    // initialize stack of node transforms to identity matrix
    auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();
    for (auto node_id : scn->nodes) 
    {
        stack.push_back(std::make_tuple(node_id, _identity_float4x4));
    }

    while (!stack.empty()) 
    {
        int              node_id;
        std::array<float, 16> xf;
        std::tie(node_id, xf)   = stack.back();
        stack.pop_back();   

        // popping from the back,  hmm does the root node need to last ?

        auto node = &m_gltf->nodes.at(node_id);

        xf = _float4x4_mul(xf, node_transform(node));   //   T-R-S-M    

        if( node->mesh == 3)
        { 
            std::cout << " node.id " << node_id << " node.mesh " << node->mesh << " node.name:" << node->name << std::endl ; 
        
            std::cout << "lxf:" ; 
            for(const auto& s: node->matrix ) std::cout << s << ' ';
            std::cout << std::endl ; 

            std::cout << "gxf: " ; 
            for(const auto& s: xf ) std::cout << s << ' ';
            std::cout << std::endl ; 
        }


        for (auto child : node->children) { stack.push_back( {child, xf} ); }
    }
}




