/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <string>
#include <iostream>
#include <fstream>

#include "YOGTF.hh"
#include "BFile.hh"

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

    for(int i=0 ; i < int(sc->nodes.size()) ; i++ )
    {
        Nd* nd = sc->nodes[i] ; 

        node_t node ; 
        node.name = nd->name ;  // pvName 
        node.mesh = nd->prIdx ;   // glTF mesh corresponds to YOG::Pr 
        node.children = nd->children ; 
        node.extras["boundary"] = nd->boundary ; 

        gltf->nodes.push_back(node) ; 
    }

    for(int i=0 ; i < int(sc->prims.size()) ; i++ )
    {
        const Pr& pr = sc->prims[i] ; 

        Mh* mh = sc->get_mesh(pr.lvIdx) ; 

        mesh_t mesh ; 
        mesh.name = mh->soName ;  

        gltf->meshes.push_back(mesh) ; 
    }
}


void TF::save(const char* path_)
{
    if( gltf == NULL ) convert();

    bool save_bin = false ;
    bool save_shaders = false ;
    bool save_images = false ;

    std::string xpath = BFile::preparePath(path_);
    const char* path = xpath.c_str();  

    std::cout << "writing " << path << std::endl ; 
    save_gltf(path, gltf, save_bin, save_shaders, save_images);

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ;
}

} // namespace



