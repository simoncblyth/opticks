#pragma once

#include <glm/glm.hpp>
#include "qef.h"

struct OctreeDrawInfo 
{
    OctreeDrawInfo() : index(-1), corners(0) {}

    int          index;
    int          corners;
    glm::vec3    position;
    glm::vec3    averageNormal;
    svd::QefData qef;

};

