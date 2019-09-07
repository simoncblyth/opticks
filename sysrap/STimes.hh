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
STimes
=======

struct to hold time measurements


**/
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API STimes {

   unsigned int count ; 
   double validate ; 
   double compile ; 
   double prelaunch ; 
   double launch ; 
   const char* _description ; 

   STimes() :
      count(0),
      validate(0),
      compile(0),
      prelaunch(0),
      launch(0),
      _description(0)
   {
   }

   const char* description(const char* msg="STimes::description");
   std::string brief(const char* msg="STimes::brief");
   std::string desc() const ;

};


