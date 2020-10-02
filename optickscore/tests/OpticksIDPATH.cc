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

// op --idpath

#include <iostream>
#include "Opticks.hh"

#include "OPTICKS_LOG.hh"

/**
OpticksIDPATH 
===============

Simple executable that dumps the directory of the 
Opticks geocache to stderr. The directory returned 
depends on the arguments provided that select a detector
DAE file and also select subsets of the geometry. 

After setting PATH use::

   args="" # potentially select 

   IDPATH="$(op --idpath 2>&1 > /dev/null)"  # capture only stderr

Formerly used::

   opticks-key2idpath(){ local dir=$(OpticksIDPATH --envkey --fatal 2>&1) ; echo $dir ; } 


**/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ;
    ok.configure();

    std::cerr << ok.getIdPath() << std::endl ; 

    return 0 ; 
}
