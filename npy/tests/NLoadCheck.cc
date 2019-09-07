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

#include <cassert>
/*

::

    NLoadCheck ~/opticksdata/gensteps/dayabay/natural/1.npy f
    NLoadCheck ~/opticksdata/gensteps/dayabay/natural/1.npy d

*/
#include "NPY.hpp"
#include "NLoad.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     NPY_LOG__ ; 

     NPYBase::setGlobalVerbose(true);

     if(argc < 2)
     {
         LOG(warning) << "expecting first argument with path to NPY array to load" ; 
         exit(0) ;
     }
      
     char* path = argv[1] ;
     char* typ = argc > 2 ? argv[2] : (char*)"f" ;  


     if( typ[0] == 'f' )
     {
         NPY<float>* af = NPY<float>::load(path) ;
         if(af == NULL)
         {
             LOG(info) << "NPY<float>::load FAILED try debugload " ; 
             af = NPY<float>::debugload(path) ; 
         }
         if(af) af->dump();
     }
     else if( typ[0] == 'd' )
     {
        NPY<double>* ad = NPY<double>::load(path);
        if(ad == NULL)
        {
             LOG(info) << "NPY<double>::load FAILED try debugload " ; 
             ad = NPY<double>::debugload(path) ; 
        }

        if(ad) ad->dump();
     }
     else
     {
         LOG(warning) << "2nd argument needs to be an f or d to pick type" ;
     }
 
     return 0 ; 
}
