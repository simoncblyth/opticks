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

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "PLOG.hh" 
#include "Pix.hh" 


void Pix::download()
{
    glPixelStorei(GL_PACK_ALIGNMENT,1); // byte aligned output https://www.khronos.org/opengl/wiki/GLAPI/glPixelStore
    glReadPixels(0,0,pwidth*pscale,pheight*pscale,GL_RGBA, GL_UNSIGNED_BYTE, pixels );

    LOG(info) 
        << desc()
        ;


}


