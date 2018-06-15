#pragma once

#include <string>
#include <vector>
#include <map>

#include "YGLTF.h"

#include "YOG.hh"
#include "YOG_API_EXPORT.hh"

/**
YOGTF
===========

* ../analytic/sc.py

**/

namespace ygltf 
{
    struct glTF_t ;   
    struct fl_gltf ;
    struct fl_mesh ;
    struct node_t ;
    struct mesh_t ;
}

namespace YOG 
{
   struct Sc ; 
}

namespace YOG 
{

struct YOG_API TF 
{
    Sc*            sc ; 
    ygltf::glTF_t* gltf ; 

    TF( Sc* sc_ ); 

    void convert();
    void save(const char* path);
};


} // namespace


