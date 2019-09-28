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

// TEST=CTreeJUNOTest om-t

/**
CTreeJUNOTest
===============

See also

* NTreeJUNOTest NTreeJUNO NSolid

**/

#include <vector>
#include "NTreeJUNO.hpp"

#include "NCSG.hpp"

#include "NNode.hpp"
#include "CMaker.hh"
#include "X4GDMLParser.hh"

#include "OPTICKS_LOG.hh"


void test_lv( int lv )
{
    nnode* a = NTreeJUNO::create(lv);   // -ve lv are rationalized
    assert( a && a->label ); 
       
    LOG(fatal) << "LV=" << lv << " label " << a->label  ; 
    LOG(error) << a->ana_desc() ; 

    NCSG::PrepTree(a); 

    G4VSolid* solid = CMaker::MakeSolid(a);

    bool refs = false ; // add pointer refs : false because already present
    std::string gdml = X4GDMLParser::ToString(solid, refs) ; 
    LOG(fatal) << gdml ; 

    const char* path = X4GDMLParser::PreparePath("$TMP/cfg4/CTreeJUNOTest", lv, ".gdml" );  
    LOG(info) << "writing gdml to " << path ; 
    X4GDMLParser::Write( solid, path, refs ); 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    int arg = argc > 1 ? atoi(argv[1]) : 0 ; 
    if( arg == 0 )
    {
        const NTreeJUNO::VI& v = NTreeJUNO::LVS ; 
        for( NTreeJUNO::VI::const_iterator it=v.begin() ; it != v.end() ; it++ ) test_lv(*it) ; 
    }
    else
    {
        test_lv(arg); 
    }

    return 0 ; 
}

