#pragma once
/**
scuda_templated.h
===================

Use F2/F4/F4 within "template<typename F>" functions 
to standin for float2/3/4 or double2/3/4 in a more flexible way. 

Inspired by:

* :google:`CUDA templated float3 or double3`
* https://forums.developer.nvidia.com/t/no-instance-of-function-template-matches-the-argument-list/19831

**/


template <typename F> struct vec2         { typedef float   Type; }; // dummy 
template <typename F> struct vec3         { typedef float   Type; }; // dummy 
template <typename F> struct vec4         { typedef float   Type; }; // dummy 
template <typename F> struct vquad        { typedef float   Type; }; // dummy 

template <>           struct vec2<float>  { typedef float2  Type; };
template <>           struct vec3<float>  { typedef float3  Type; };
template <>           struct vec4<float>  { typedef float4  Type; };
template <>           struct vquad<float> { typedef quad    Type; };

#ifdef WITH_SCUDA_DOUBLE
template <>           struct vec2<double> { typedef double2 Type; };
template <>           struct vec3<double> { typedef double3 Type; };
template <>           struct vec4<double> { typedef double4 Type; };
template <>           struct vquad<double> { typedef dquad Type; };
#endif

#define F2 typename vec2<F>::Type
#define F3 typename vec3<F>::Type
#define F4 typename vec4<F>::Type
#define Q4 typename vquad<F>::Type


