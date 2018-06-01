// externals/yoctogl/yocto-gl/tests/ygltf_reader.cc

#include <string>
#include <iostream>
#include <fstream>

#include "YGLTF.h"

using ygltf::node_t ; 
using ygltf::scene_t ; 
using ygltf::glTF_t ; 


std::unique_ptr<glTF_t> make_gltf()
{
    // NB : there is no checking can easily construct non-sensical gltf 
    //      as shown below 

    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<scene_t>& scenes = gltf->scenes ; 

    node_t a, b, c  ; 
    a.children = {2} ;   
    b.children = {3} ;   
    c.children = {} ;   

    nodes = { a, b, c } ;

    scene_t sc ;    // scene references nodes by index
    sc.nodes = {1,2,3} ;

    scenes = {sc} ;  

    return gltf ; 
}


int main(int argc, char** argv)
{
    auto gltf = make_gltf() ; 

    std::string path = "/tmp/UseYoctoGL_Write.gltf" ; 
    bool save_bin = false ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    save_gltf(path, gltf.get(), save_bin, save_shaders, save_images);

    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 
    return 0 ; 
}
