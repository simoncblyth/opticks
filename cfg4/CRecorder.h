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


#include "CFG4_API_EXPORT.hh"

struct CFG4_API short4 
{  
   short x ; 
   short y ; 
   short z ; 
   short w ; 
};

struct CFG4_API ushort4 
{  
   unsigned short x ; 
   unsigned short y ; 
   unsigned short z ; 
   unsigned short w ; 
};

union CFG4_API hquad
{   
   short4   short_ ;
   ushort4  ushort_ ;
};  

struct CFG4_API char4
{
   char x ; 
   char y ; 
   char z ; 
   char w ; 
};

struct CFG4_API uchar4
{
   unsigned char x ; 
   unsigned char y ; 
   unsigned char z ; 
   unsigned char w ; 
};

union CFG4_API qquad
{   
   char4   char_   ;
   uchar4  uchar_  ;
};  

union CFG4_API uifchar4
{
   unsigned int u ; 
   int          i ; 
   float        f ; 
   char4        char_   ;
   uchar4       uchar_  ;
};



