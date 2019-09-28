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

#include "BConfig.hh"
#include "PLOG.hh"
#include "NSnapConfig.hpp"

const plog::Severity NSnapConfig::LEVEL = debug ; 


NSnapConfig::NSnapConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    steps(10),
    fmtwidth(5),
    eyestartx(-0.f),   // -ve zero on startx,y,z indicates leave asis, see OpTracer::snap
    eyestarty(-0.f),
    eyestartz(0.f),
    eyestopx(-0.f),
    eyestopy(-0.f),
    eyestopz(1.f),
    prefix("snap"),
    postfix(".ppm")
{
    LOG(LEVEL)
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    // TODO: incorp the help strings into the machinery and include in dumping 

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("steps", &steps );
    bconfig->addInt("fmtwidth", &fmtwidth );

    bconfig->addFloat("eyestartx", &eyestartx );
    bconfig->addFloat("eyestopx", &eyestopx );

    bconfig->addFloat("eyestarty", &eyestarty );
    bconfig->addFloat("eyestopy", &eyestopy );

    bconfig->addFloat("eyestartz", &eyestartz );
    bconfig->addFloat("eyestopz", &eyestopz );

    bconfig->addString("prefix", &prefix );
    bconfig->addString("postfix", &postfix );

    bconfig->parse();
}

void NSnapConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

std::string NSnapConfig::SnapIndex(unsigned index, unsigned width)
{
    std::stringstream ss ;
    ss 
       << std::setw(width) 
       << std::setfill('0')
       << index 
       ;
    return ss.str();
}


std::string NSnapConfig::getSnapName(unsigned index)
{
    std::stringstream ss ;
    ss 
       << prefix 
       << SnapIndex(index, fmtwidth)
       << postfix 
       ; 

    return ss.str();
}

std::string NSnapConfig::desc() const 
{
    return bconfig->desc(); 
}
