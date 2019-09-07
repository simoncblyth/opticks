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

#include <iostream>
#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "BConfig.hh"


struct DemoConfig : BConfig
{
    DemoConfig(const char* cfg, const char* kvdelim);

    int red ; 
    int green ; 
    int blue ; 

    float cyan ; 
    std::string magenta ; 

};

DemoConfig::DemoConfig(const char* cfg_, const char* kvdelim)
   : 
   BConfig(cfg_, ',', kvdelim),

   red(0),
   green(0),
   blue(0),
   cyan(1.f),
   magenta("hello")
{
   addInt("red",   &red); 
   addInt("green", &green); 
   addInt("blue",  &blue); 
   addFloat("cyan",  &cyan); 
   addString("magenta",  &magenta); 

   parse();
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG__ ; 

    DemoConfig cfg("red=1,green=2,blue=3,cyan=1.5,magenta=purple", "=");
    cfg.dump();


    std::cout << " cyan " << cfg.cyan << std::endl ; 
    std::cout << " magenta " << cfg.magenta << std::endl ; 


    DemoConfig cfg2("red:1,green:2,blue:3,cyan:1.5,magenta:purple",":");
    cfg2.dump();





    return 0 ; 
}

