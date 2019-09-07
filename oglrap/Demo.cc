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

#include <cstdio>
#include <cassert>

#include "GVector.hh"

#include "Demo.hh"

/*
 
           y ^
             |
             R
             | 
             |
      -------|--------> x
             |
             |
        B    |    G 
 

*/


#define XOFF  100.0f
#define YOFF  100.0f
#define ZOFF  100.0f
#define VAL    10.0f

const float Demo::pvertex[] = {
   0.0f+XOFF,  VAL+YOFF,  0.0f+ZOFF,
    VAL+XOFF, -VAL+YOFF,  0.0f+ZOFF,
   -VAL+XOFF, -VAL+YOFF,  0.0f+ZOFF
};

const float Demo::pcolor[] = {
  1.0f, 0.0f,  0.0f,
  0.0f, 1.0f,  0.0f,
  0.0f, 0.0f,  1.0f
};

const float Demo::pnormal[] = {
  0.0f, 0.0f,  1.0f,
  0.0f, 0.0f,  1.0f,
  0.0f, 0.0f,  1.0f
};

const float Demo::ptexcoord[] = {
  0.0f, 0.0f,
  1.0f, 0.0f,
  1.0f, 1.0f
};


const unsigned int Demo::pindex[] = {
      0,  1,  2
};


Demo::Demo() : GMesh(0, (gfloat3*)&pvertex[0],3, (guint3*)&pindex[0],1, (gfloat3*)&pnormal[0], (gfloat2*)&ptexcoord[0]) 
{
    setColors( (gfloat3*)&pcolor[0] );
}

Demo::~Demo()
{
}


