#pragma once

/**
OXPPNS
========

OptiX and its C++ interface headers

**/




#include "OXRAP_PUSH.hh"
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)

#include "OXRAP_POP.hh"



