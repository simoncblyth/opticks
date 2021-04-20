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

#include <sstream>

#include "SSys.hh"
#include "BFile.hh"
#include "BConfig.hh"
#include "PLOG.hh"
#include "NFlightConfig.hpp"

const plog::Severity NFlightConfig::LEVEL = PLOG::EnvLevel("NFlightConfig","DEBUG") ; 

NFlightConfig::NFlightConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    width(5),
    scale0(1.f), 
    scale1(1.f),
    idir("/tmp"),
    prefix("flight"),
    ext(".jpg"),
    framelimit(SSys::getenvint("OPTICKS_FLIGHT_FRAMELIMIT",3))
{
    LOG(LEVEL) << cfg ; 

    bconfig->addInt("width", &width );
    bconfig->addFloat("scale0", &scale0 );
    bconfig->addFloat("scale1", &scale1 );

    bconfig->addString("idir",   &idir );
    bconfig->addString("prefix", &prefix );
    bconfig->addString("ext",    &ext );   
    bconfig->addInt("framelimit", &framelimit );

    bconfig->parse();
}

void NFlightConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

std::string NFlightConfig::getFrameName(int index, const char* override_prefix) const 
{
    const char* pfx = override_prefix ? override_prefix : prefix.c_str()  ; 
    return BFile::MakeName(index, width, pfx, ext.c_str() ); 
}

const char* NFlightConfig::getFramePath(const char* dir, const char* reldir, int index, const char* override_prefix) const 
{
    std::string name = getFrameName(index, override_prefix) ; 
    bool create = true ; 
    std::string path = BFile::preparePath(dir ? dir : "$TMP", reldir, name.c_str(), create);  
    return strdup(path.c_str()); 
}

std::string NFlightConfig::desc() const 
{
    return bconfig->desc() ;
}
