#include <string>
#include <iostream>
#include <fstream>

#define TEMPORARY 1
#ifdef TEMPORARY
#include "NNode.hpp"
//#include "GMesh.hh"
#endif

#include "YOGTF.hh"

using ygltf::glTF_t ; 
using ygltf::scene_t ; 
using ygltf::node_t ; 
using ygltf::mesh_t ; 
using ygltf::mesh_primitive_t ; 
using ygltf::buffer_t ; 
using ygltf::bufferView_t ; 
using ygltf::accessor_t ; 

namespace YOG 
{

TF::TF( Sc* sc_ )
    :
    sc(sc_),
    gltf(NULL)
{   
}

void TF::convert()
{
    gltf = new glTF_t() ;

    scene_t scene ; 
    scene.nodes = { sc->root } ; 
    gltf->scenes = { scene } ;

    for(int i=0 ; i < sc->nodes.size() ; i++ )
    {
        Nd* nd = sc->nodes[i] ; 

        node_t node ; 
        node.name = nd->name ;  // pvName 
        node.mesh = nd->soIdx ; 
        node.children = nd->children ; 
        node.extras["boundary"] = nd->boundary ; 

        gltf->nodes.push_back(node) ; 
    }

    for(int i=0 ; i < sc->meshes.size() ; i++ )
    {
        Mh* mh = sc->meshes[i] ; 

        mesh_t mesh ; 
        mesh.name = mh->soName ;  // 

#ifdef TEMPORARY
        mesh.extras["csg.desc"] = mh->csg ? mh->csg->tag() : "-" ; 
     //   mesh.extras["mesh.desc"] = mh->mesh ? mh->mesh->desc() : "-" ; 
#endif

        gltf->meshes.push_back(mesh) ; 
    }
}


void TF::save(const char* path)
{
    if( gltf == NULL ) convert();

    bool save_bin = false ;
    bool save_shaders = false ;
    bool save_images = false ;

    std::cout << "writing " << path << std::endl ;
    save_gltf(path, gltf, save_bin, save_shaders, save_images);

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ;
}

} // namespace



