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

#pragma once

/**
OXPPNS
========

OptiX and its C++ interface headers

**/




#include "OXRAP_PUSH.hh"

#include <optix.h>
#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)

#if OPTIX_VERSION_MAJOR >= 7

#else
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#endif

#include "OXRAP_POP.hh"



