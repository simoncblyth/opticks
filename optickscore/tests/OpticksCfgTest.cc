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

// TEST=OpticksCfgTest om-t

#include "NSnapConfig.hpp"

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    const char* argforce = "--nogdmlpath" ; 

    Opticks* ok = new Opticks(argc, argv, argforce);

    BCfg* cfg  = new BCfg("umbrella", false) ;

    BCfg* ocfg = new OpticksCfg<Opticks>("opticks", ok,false);

    cfg->add(ocfg);


    int    _argc = ok->getArgc(); 
    char** _argv = ok->getArgv(); 

    cfg->commandline(_argc, _argv);

    std::string desc = cfg->getDescString();

    LOG(info) << "desc... " << desc ;

    LOG(info) << "ocfg... "  ;

    ocfg->dump("dump");

    LOG(info) << "sc... "  ;

    NSnapConfig* sc = ok->getSnapConfig();
    sc->dump("SnapConfig");


    assert( ok->isNoGDMLPath() ); 


    //const std::string& csgskiplv = ocfg->getCSGSkipLV(); 
    //LOG(info) << " csgskiplv " << csgskiplv ; 


    LOG(info) << "DONE "  ;



    return 0 ; 
}
