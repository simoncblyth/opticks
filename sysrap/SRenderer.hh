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
SRenderer : Protocol base
============================

Duties of the *render* method
--------------------------------

1. access eye-look-up and camera parameters from Composition and update GPU context accordingly 
2. do the launch, recording the time
3. return the launch time    

Duties of the *snap* method
-----------------------------

1. download the frame buffer into CPU side pixels buffer
2. annotate pixels buffer with top/bottom lines 
3. save pixels buffer to to file at the path provided

**/

class SRenderer {
   public:
      virtual double render() = 0 ;
      virtual void snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height ) = 0 ;  

};


