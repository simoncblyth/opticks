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

#include "Index.hpp"
#include <cassert>
#include "PLOG.hh"



// see ggeo-/GItemIndexTest  op --gitemindex " 

int main(int argc , char** argv )
{
   PLOG_(argc, argv);

   const char* reldir = NULL ; 
   Index idx("IndexTest", reldir);
   idx.add("red",1);
   idx.add("green",2);
   idx.add("blue",3);

   assert(idx.getIndexSource("green") == 2 );

   int* ptr = idx.getSelectedPtr();
  
   for(unsigned i=0 ; i < idx.getNumKeys() ; i++ )
   { 
      *ptr = i ; 
      LOG(info) << std::setw(4) << i << " " << idx.getSelectedKey() ; 
   }


   return 0 ; 
}

