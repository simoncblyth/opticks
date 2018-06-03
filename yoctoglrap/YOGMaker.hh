#pragma once

#include <vector>

#include "YGLTF.h"
#include "YOG_API_EXPORT.hh"

struct YOGGeometry ;

class YOG_API YOGMaker 
{
   public:

    static std::unique_ptr<ygltf::glTF_t> make_gltf(const YOGGeometry& geom );

};





