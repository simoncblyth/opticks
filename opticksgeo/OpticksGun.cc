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

#include "OpticksGun.hh"
#include "OpticksHub.hh"
#include "Opticks.hh"

#include "PLOG.hh"

OpticksGun::OpticksGun(OpticksHub* hub)
   :
   m_hub(hub),
   m_ok(hub->getOpticks())
{
    init();
}

void OpticksGun::init()
{
}

std::string OpticksGun::getConfig()
{
    std::string config = m_ok->hasOpt("g4gun") ? m_ok->getG4GunConfig() : "" ; 

    if(config.size() == 0)
    {
        int itag = m_ok->getEventITag();
        LOG(warning) << "OpticksGun::getConfig"
                     << " using assignTagDefault as g4gunconfig is blank "
                     << " itag : " << itag 
                     << " config : " << config 
                     ; 
         assignTagDefault(config, itag);  
    }

    //assert(config.size() > 0);

    return config ; 
}


void OpticksGun::assignTagDefault(std::string& config, int itag)
{
    if( itag == 1 )
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=1000.0_"
    "number=1_")
    ;  // mm,ns,MeV 

    else if(itag == 100)
         config.assign(
    "comment=default-config-comment-without-spaces-_"
    "particle=mu-_"
    "frame=3153_"
    "position=0,0,-1_"
    "direction=0,0,1_"
    "polarization=1,0,0_"
    "time=0.1_"
    "energy=100000.0_"
    "number=1_")
    ;  // mm,ns,MeV 
}

 
