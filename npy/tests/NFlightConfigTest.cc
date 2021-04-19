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

NFlightConfigTest --snapconfig "steps=10,eyestartz=-1,eyestopz=1" 


**/

#include "OPTICKS_LOG.hh"
#include "NFlightConfig.hpp"
#include "S_get_option.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* fallback = "prefix=frame_,ext=.ppm,scale0=1,scale1=10" ; 
    std::string flightconfig = get_option<std::string>(argc, argv, "--flightconfig", fallback ) ;
    // need to use this because cannot use Opticks at this level 

    NFlightConfig cfg(flightconfig.c_str());
    cfg.dump();
  
    std::cout << cfg.getFrameName(0) << std::endl ; 

    return 0 ; 
}
