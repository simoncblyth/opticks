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

#include "BPropNames.hh"
#include "BFile.hh"
#include "BTxt.hh"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <climits>


const char* TMPDIR = "$TMP/boostrap/BPropNamesTest" ; 

int main(int, char**)
{

    std::string p = BFile::FormPath(TMPDIR, "BPropNamesTest.txt" ); 
    const char* lib = p.c_str();  // absolute path mode for testing 


    BTxt txt(lib);
    txt.addLine("red");
    txt.addLine("green");
    txt.addLine("blue");
    txt.write();


    BPropNames pn(lib);

    for(unsigned int i=0 ; i < pn.getNumLines() ; i++)
    {
        std::string line = pn.getLine(i) ;
        unsigned int index = pn.getIndex(line.c_str());

        std::cout << " i  " << std::setw(3) << i 
                  << " ix " << std::setw(3) << index
                  << " line " << line
                  << std::endl 
                  ; 

         assert(i == index);
    }

    assert( pn.getIndex("THIS_LINE_IS_NOT_PRESENT") == UINT_MAX );

    return 0 ; 
}
