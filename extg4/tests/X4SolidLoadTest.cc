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
#include "X4.hh"
#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "NCSGList.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    
    int lvIdx = SSys::getenvint("LV",65) ;
    std::string csgpath = BFile::FormPath( X4::X4GEN_DIR, BStr::concat("x", BStr::utoa(lvIdx,3, true), NULL)) ;   

    LOG(info) << " lvIdx " << lvIdx << " csgpath " << csgpath ; 

    NCSGList* ls = NCSGList::Load(csgpath.c_str());  
    if(!ls) LOG(error) << "failed to load " << csgpath ; 

    return 0 ; 
}
