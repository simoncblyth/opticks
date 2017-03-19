#include <cmath>
#include <cassert>
#include <cmath>
#include <cstring>

#include <boost/math/constants/constants.hpp>

#include "OpticksCSG.h"

#include "NPrism.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"


nprism::nprism(float apex_angle_degrees, float height_mm, float depth_mm, float fallback_mm)
{
    param.x = apex_angle_degrees  ;
    param.y = height_mm  ;
    param.z = depth_mm  ;
    param.w = fallback_mm  ;
}

nprism::nprism(const nvec4& param_)
{
    param = param_ ;
}

float nprism::height()
{
    return param.y > 0.f ? param.y : param.w ; 
}
float nprism::depth()
{
    return param.z > 0.f ? param.z : param.w ; 
}
float nprism::hwidth()
{
    float pi = boost::math::constants::pi<float>() ;
    return height()*tan((pi/180.f)*param.x/2.0f) ;
}


void nprism::dump(const char* msg)
{
    param.dump(msg);
}

nbbox nprism::bbox()
{
    float h  = height();
    float hw = hwidth();
    float d  = depth();

    nbbox bb ;
    bb.min = {-hw,0.f,-d/2.f } ;
    bb.max = { hw,  h, d/2.f } ;

    return bb ; 
}


npart nprism::part()
{
    // hmm more dupe of hemi-pmt.cu/make_prism
    // but if could somehow make vector types appear 
    // the same could use same code with CUDA ?

    nbbox bb = bbox();

    npart p ; 
    p.zero();            
    p.setParam(param) ; 
    p.setTypeCode(CSG_PRISM); 
    p.setBBox(bb);

    return p ; 
}


