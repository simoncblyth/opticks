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
    flight("RoundaboutXY"),
    ext(".jpg"),
    period(4),
    framelimit(3),
    framelimit_override(SSys::getenvint("OPTICKS_FLIGHT_FRAMELIMIT",0))
{
    LOG(LEVEL) << cfg ; 

    bconfig->addInt("width", &width );
    bconfig->addFloat("scale0", &scale0 );
    bconfig->addFloat("scale1", &scale1 );

    bconfig->addString("flight", &flight );
    bconfig->addString("ext",    &ext );   
    bconfig->addInt("period",    &period);
    bconfig->addInt("framelimit", &framelimit );

    bconfig->parse();
}

const char*  NFlightConfig::getCfg() const 
{
    return bconfig->cfg ; 
}

void NFlightConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

/**
NFlightConfig::getFrameLimit
------------------------------

Returns value parsed from --flightconfig option unless the 
overriding envvar OPTICKS_FLIGHT_FRAMELIMIT is defined and greated than zero.

**/

unsigned NFlightConfig::getFrameLimit() const 
{
    return framelimit_override > 0 ? framelimit_override : framelimit ; 
}

/**
NFlightConfig::getFrameName
-----------------------------

Index -1 returns a printf format string to be filled with a single integer.

**/

std::string NFlightConfig::getFrameName(const char* prefix, int index) const 
{
    return BFile::MakeName(index, width, prefix, ext.c_str() ); 
}

std::string NFlightConfig::desc() const 
{
    return bconfig->desc() ;
}

