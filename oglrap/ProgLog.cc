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

#include <cstring>
#include <GL/glew.h>

#include "PLOG.hh"
#include "ProgLog.hh"

const char* ProgLog::NO_FRAGMENT_SHADER = "Validation Failed: Program does not contain fragment shader. Results will be undefined.\n" ;

ProgLog::ProgLog(int id_) : id(id_), length(0) 
{
    glGetProgramInfoLog (id, MAX_LENGTH, &length, log);
}

bool ProgLog::is_no_frag_shader() const 
{
    return strcmp(NO_FRAGMENT_SHADER, log) == 0 ;
}

void ProgLog::dump(const char* msg)
{
    LOG(info) << msg ; 
    printf ("ProgLog::dump id %u:\n[%s]", id, log);
}



