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

#include "BFile.hh"
#include "BConfig.hh"
#include "PLOG.hh"
#include "NSnapConfig.hpp"

const plog::Severity NSnapConfig::LEVEL = PLOG::EnvLevel("NSnapConfig","DEBUG") ; 

const float NSnapConfig::NEGATIVE_ZERO = -0.f ; 

NSnapConfig::NSnapConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    steps(10),
    fmtwidth(5),
    eyestartx(NEGATIVE_ZERO),   // -ve zero on startx,y,z indicates leave asis, see OpTracer::snap
    eyestarty(NEGATIVE_ZERO),
    eyestartz(0.f),
    eyestopx(NEGATIVE_ZERO),
    eyestopy(NEGATIVE_ZERO),
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

std::string NSnapConfig::SnapIndex(int index, unsigned width) // static 
{
    std::stringstream ss ;
    ss 
        << std::setw(width) 
        << std::setfill('0')
        << index 
        ;
    return ss.str();
}


std::string NSnapConfig::getSnapName(int index) const 
{
    std::string blank(""); 
    std::stringstream ss ;
    ss 
       << prefix 
       << ( index > -1 ? SnapIndex(index, fmtwidth) : blank )
       << postfix 
       ; 

    return ss.str();
}

const char* NSnapConfig::getSnapPath(const char* dir, const char* reldir, int index) const 
{
    std::string name = getSnapName(index) ; 
    bool create = true ; 
    std::string path = BFile::preparePath(dir ? dir : "$TMP", reldir, name.c_str(), create);  
    return strdup(path.c_str()); 
}


std::string NSnapConfig::desc() const 
{
    std::stringstream ss ; 
    ss 
       << bconfig->desc() 
       << " [change .cfg with --snapconfig] "
       ;

    std::string s = ss.str(); 
    return s ; 
}
