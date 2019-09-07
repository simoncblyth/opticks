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
/*

Currently:

    q0 : param       <-- primitive use
    q1 : xy:prim        <-- primitive use
    ------------------------------------------------
    q2 : xyz:bbox 
    q3 : xyz:bbox     

Aiming for:

    q0 : param    <-- primitive use
    q1 : param1   <-- primitive use
    ------------------------------------------------
    q2 :  
    q3 :     


Hmm supporting partlist with its bbox makes it 
difficult to rejig the layout, so for now just shift the id to w

2017-04-16 16:44:17.307 INFO  [513893] [GParts::dump@857] GParts::dump ni 4
     0.0000      0.0000      0.0000   1000.0000 
     0.0000       0 <-id       123 <-bnd       0.0000  bn Rock//perfectAbsorbSurface/Vacuum 
     0.0000      0.0000      0.0000           6 (box) TYPECODE 
     0.0000      0.0000      0.0000           0 (nodeIndex) 

     0.0000      0.0000      0.0000      0.0000 
     0.0000       1 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000           2 (intersection) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 

     0.0000      0.0000      0.0000    500.0000 
     0.0000       2 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000           5 (sphere) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 

     0.0000      0.0000      1.0000      0.0000 
  -100.0000       3 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000          13 (slab) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 



Argh getting a clash for convexpolyhedron between bbmax.x and the gtransform


2017-05-01 19:16:55.450 INFO  [3453366] [GParts::dump@971] GParts::dump ni 2
     0.0000      0.0000      0.0000      0.0000 
     0.0000      0.0000     123 <-bnd        0 <-INDEX    bn Vacuum///GlassSchottF2 
  -201.0000   -201.0000   -201.0000          19 (convexpolyhedron) TYPECODE 
     0.0000    201.0000    201.0000           0 (nodeIndex) 
*/


enum { PARAM_J  = 0, PARAM_K   = 0  };  // q0.f.xyzw


enum { PARAM1_J   = 1, PARAM1_K   = 0  };  // q1.f.xyzw
enum { INDEX_J    = 1, INDEX_K    = 3  };   // q1.u.w
enum { BOUNDARY_J = 1, BOUNDARY_K = 2  };   // q1.u.z


enum { BBMIN_J     = 2,  BBMIN_K     = 0   }; // q2.f.xyz
enum { TYPECODE_J  = 2,  TYPECODE_K  = 3  };  // q2.u.w


//enum { GTRANSFORM_J = 3,  GTRANSFORM_K = 0  };   // q3.u.x
enum { GTRANSFORM_J = 3,  GTRANSFORM_K = 3  };   // q3.u.w     (as it seems nodeIndex not used GPU side)
enum { BBMAX_J      = 3,  BBMAX_K = 0   };       // q3.f.xyz 
enum { TRANSFORM_J  = 3,  TRANSFORM_K = 3  };    // q3.u.w   : only used for CSG operator nodes in input serialization buffer
enum {  NODEINDEX_J = 3,  NODEINDEX_K = 3  };    // q3.u.w   <-- WHAT USES THIS GPU side ?



