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

// om-; TEST=GItemIndex2Test om-t

#include <string>
#include <iostream>
#include <cassert>

#include "Opticks.hh"
#include "GItemIndex.hh"
#include "GItemList.hh"
#include "OPTICKS_LOG.hh"

const char* GITEMINDEX = "GItemIndex" ;
const char* GMESHLIB_INDEX = "MeshIndex" ;

/**
GItemIndex2Test
================

Loads meshindex GItemIndex from idpath and creates a GItemList
When the macro WRITE_MESHNAMES_TO_GEOCACHE is defined
this will then save the meshnames GItemList.

Doing this allows the old geocache to work with the
new GMeshLib which has moved from GItemIndex to GItemList. 
In so doing this avoids a large number of test failures.


**/

// #define WRITE_MESHNAMES_TO_GEOCACHE 1


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* idpath = ok.getIdPath() ;
    GItemIndex* m_meshindex = GItemIndex::load(idpath, GITEMINDEX, GMESHLIB_INDEX ) ;

    m_meshindex->dump();

    GItemList* m_meshnames = new GItemList("GMeshLib", "GItemList"); 

    unsigned num_mesh = m_meshindex->getNumItems();
    for(unsigned i=0 ; i < num_mesh ; i++)
    {
         const char* name = m_meshindex->getNameSource(i) ;  // hmm assumes source index is 0:N-1 which is not gauranteed
/*
         std::cout 
             << std::setw(3) << i 
             << std::setw(50) << name
             << std::endl 
             ;
*/
         m_meshnames->add(name); 
    }  

    LOG(info) << " NumUniqueKeys " << m_meshnames->getNumUniqueKeys()  ; 
    assert( m_meshnames->getNumUniqueKeys() == num_mesh ); 
    m_meshnames->dump() ; 

#ifdef WRITE_MESHNAMES_TO_GEOCACHE
    // naughty writing into geocache from a test 
    m_meshnames->save(idpath);  
    LOG(info) << idpath ; 
#endif

    return 0 ;
}
