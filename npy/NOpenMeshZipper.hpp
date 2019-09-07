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

template <typename T> struct NOpenMesh ; 

/*
NOpenMeshZipper
==================

Each side has a frontier ribbon mesh::

      +---------+---------+---------+---          outer loop (SDF other >  0)   
       \       / \       / \       / \
      . * . . * . * . . *. .*. . .* . * . . . .   analytic frontier (SDF other = 0 )       
         \   /     \   /     \   /     \
          \ /       \ /       \ /       \
      -----+---------+---------+---------+        inner loop (SDF other < 0) 


CSG Union requires zippering of

* inner loops of both sides with the frontier vertices in the middle 



*/ 

template <typename T>
struct NPY_API  NOpenMeshZipper
{
    typedef typename T::Point              P ; 

    NOpenMeshZipper(
                    const NOpenMesh<T>& lhs, 
                    const NOpenMesh<T>& rhs
                   );



    void init();

    void dump();
    void dump_boundary(int index, const NOpenMeshBoundary<T>& loop, const char* msg);


    const NOpenMesh<T>&  lhs ; 
    const NOpenMesh<T>&  rhs ; 
}; 




