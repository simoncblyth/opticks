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

#include "NPY_API_EXPORT.hh"
#include <string>

/**
NGrid
======

* (row, column) grid of T struct pointers, initally all NULL
* set/get pointers in the grid, battleship style
* string display assuming only that the T struct has a "label" 
  member that provides brief identification eg 3 characters only.

**/

template <typename T>
struct NPY_API NGrid 
{
   NGrid(unsigned nr_, unsigned nc_, unsigned width_=4, const char* unset_="", const char* rowjoin_="\n\n" );
   ~NGrid();

   void init();
   void clear();

   unsigned idx(unsigned r, unsigned c) const  ; 
   void     set(unsigned r, unsigned c, const T* ptr);
   const T* get(unsigned r, unsigned c) const ; 

   std::string desc() ;

   unsigned    nr ;  
   unsigned    nc ;  
   unsigned    width ; 
   const char* unset ;
   const char* rowjoin ;

   const T**   grid ;   // linear array of pointers to T 
   
   // Judged that the hassles of 2d arrays are not worth the bother 
   // for the minor benefit of 2d indexing,
   // when can trivially compute a linear index : in a way that 
   // easily scales to 3,4,5,.. dimensions.

};
