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
#include "OPTICKS_LOG.hh"



// see ggeo-/GItemIndexTest  op --gitemindex " 

int main(int argc , char** argv )
{
   OPTICKS_LOG(argc, argv);

   const char* reldir = NULL ; 

   bool sort = true ;  

   std::cout << "#0" << std::endl ;  
   Index idx("IndexTest", reldir);
   std::cout << "#1" << std::endl ;  
   idx.add("red",1, sort);
   std::cout << "#2" << std::endl ;  
   idx.add("green",2, sort);
   std::cout << "#3" << std::endl ;  
   idx.add("blue",3, sort );
   std::cout << "#4" << std::endl ;  

   assert(idx.getIndexSource("green") == 2 );

   int* ptr = idx.getSelectedPtr();
  
   for(unsigned i=0 ; i < idx.getNumKeys() ; i++ )
   { 
      *ptr = i ; 
      LOG(info) << std::setw(4) << i << " " << idx.getSelectedKey() ; 
   }


   return 0 ; 
}

