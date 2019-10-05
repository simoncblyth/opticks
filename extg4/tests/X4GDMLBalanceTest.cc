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

/**

::

    X4GDMLBalanceTest $HOME/Opticks_install_guide/x375.gdml
    X4GDMLBalanceTest $HOME/Opticks_install_guide/x376.gdml


**/

#include "X4GDMLParser.hh"
#include "X4Solid.hh"
#include "Opticks.hh"
#include "BFile.hh"
#include "NNode.hpp"
#include "NCSG.hpp"
#include "NTreeProcess.hpp"
#include "NSceneConfig.hpp"


#include "OPTICKS_LOG.hh"


int main( int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure();
    
    const char* path = argc > 1 ? argv[1] : "$HOME/Opticks_install_guide/x375.gdml" ; 

    if(!BFile::ExistsFile(path)) 
    {
        LOG(error) << " NON EXISTING path " << path  ; 
        return 0 ; 
    }

    int offset = -2 ; 
    const G4VSolid* solid = X4GDMLParser::Read(path, offset) ;

    if( solid == NULL ) 
    {
        LOG(error) 
            << " NULL SOLID " 
            << " path " << path 
            << " offset " << offset
            ; 
        return 0 ; 
    }

    LOG(info) << " solid " << solid ; 

    unsigned soIdx = 0 ; 
    unsigned lvIdx = 0 ; 

    nnode* raw = X4Solid::Convert(solid, &ok); 
    LOG(info)  << raw->ana_brief() ;  
    //LOG(info)  << raw->ana_desc() ;  

    bool balance = false ; 
    nnode* use = balance ? NTreeProcess<nnode>::Process(raw, soIdx, lvIdx) : raw  ; 

    LOG(info)  << "use\n" << use->ana_desc() ;  

    const NSceneConfig* config = NULL ; 
    NCSG* tree = NCSG::Adopt( use, config, soIdx, lvIdx) ; 

    LOG(info) 
        << " tree " << tree
        ;  

    return 0 ; 
}
