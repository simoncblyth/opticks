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

#ifdef __clang__
#pragma clang diagnostic push
// gleq.h:75:9: warning: anonymous types declared in an anonymous union are an extension
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif


#define NEWGLEQ 1

#ifdef NEWGLEQ
#include "gleq.h"
#else
#include "old_gleq.h"
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif



