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

#include <string>
#include "BList.hh"

const char* ini = "$TMP/boostrap/BListTest/BListTest.ini" ;
const char* json = "$TMP/boostrap/BListTest/BListTest.json" ;
typedef std::pair<std::string, unsigned int> SU ; 

void test_saveList()
{
   std::vector<SU> vp ; 
   vp.push_back(SU("hello",1));  
   vp.push_back(SU("hello",2));  // this replaces the first hello
   vp.push_back(SU("world",3));

   BList<std::string,unsigned int>::save(&vp, ini);
   BList<std::string,unsigned int>::save(&vp, json);
}

void test_loadList()
{
   std::vector<SU> vp ; 
   BList<std::string,unsigned int>::load(&vp, ini);
   BList<std::string,unsigned int>::dump(&vp, "loadList.ini");

   vp.clear(); 
   BList<std::string,unsigned int>::load(&vp, json);
   BList<std::string,unsigned int>::dump(&vp, "loadList.json");
}



int main()
{
    test_saveList();
    test_loadList();
    return 0 ; 
}

