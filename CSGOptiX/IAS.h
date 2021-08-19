#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "AS.h"

/**
IAS
===

Static Build methods from glm::mat4 or float vector which must 
contain a multiple of 16 values.

For OpenGL convenience Opticks uses transforms like this
with the transform in the last four slots in memory::

    1  0  0  a
    0  1  0  b
    0  0  1  c 
    tx ty tz d

OptiX7 needs 3*4 floats, so transpose first::

    1  0  0  tx
    0  1  0  ty
    0  0  1  tz
    a  b  c  d

The a,b,c,d are "spare" slots used to carry unsigned_as_float identity info.

**/

struct IAS : public AS
{
    std::vector<glm::mat4>  trs ; 
    CUdeviceptr             d_instances ;   
};


