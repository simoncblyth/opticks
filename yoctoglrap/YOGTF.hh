#pragma once

#include <string>
#include <vector>
#include <map>

#include "YGLTF.h"

#include "YOG.hh"
#include "YOG_API_EXPORT.hh"

/**
YOG::TF
===========

* creates simple subset of glTF, for creation of full renderable glTF see YOGMaker
* following ../analytic/sc.py
* creation of gltf is deferred until convert 

**/

namespace ygltf 
{
    struct glTF_t ;   
}

namespace YOG 
{

struct Sc ; 

struct YOG_API TF 
{
    Sc*            sc ; 
    ygltf::glTF_t* gltf ; 

    TF( Sc* sc_ ); 

    void convert();
    void save(const char* path);
};

} // namespace


