#pragma once
#include "NPY_API_EXPORT.hh"

namespace ygltf 
{
    struct glTF_t ;  
}

class NPY_API NScene 
{
    public:
        static NScene* load(const char* base, const char* name="scene.gltf");
    public:
        NScene( ygltf::glTF_t* gltf );
        void walk();
    private:
        ygltf::glTF_t* m_gltf ;  

};

