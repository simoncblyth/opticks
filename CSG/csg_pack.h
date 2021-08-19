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

/**0
pack.h
========

* * https://bitbucket.org/simoncblyth/opticks/src/master/optixrap/cu/pack.h

.. contents:: Table of Contents
   :depth: 2

0**/


/**1
pack.h : PACK4 Macro
---------------------

Packs 4 8bit integers into 32 bits  

1**/

#define PACK4( a, b, c, d)   ( \
       (( (a) & 0xff ) <<  0 ) | \
       (( (b) & 0xff ) <<  8 ) | \
       (( (c) & 0xff ) << 16 ) | \
       (( (d) & 0xff ) << 24 ) \
                             )

/**2
pack.h : UNPACK4_0/1/2/3 Macros
--------------------------------

Returns 8 bit constituent from the packed 32 bits. 

2**/

#define UNPACK4_0( packed ) (  ((packed) >>  0) & 0xff )
#define UNPACK4_1( packed ) (  ((packed) >>  8) & 0xff )
#define UNPACK4_2( packed ) (  ((packed) >> 16) & 0xff )
#define UNPACK4_3( packed ) (  ((packed) >> 24) & 0xff )


