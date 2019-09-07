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


#include "PLOG.hh"
#include "BConfig.hh"


struct DemoConfig 
{
    DemoConfig(const char* cfg);

    struct BConfig* bconfig ; 
    void dump() const  ;
    
    int red ; 
    int green ; 
    int blue ; 
};

DemoConfig::DemoConfig(const char* cfg)
   : 
   bconfig(new BConfig(cfg)),

   red(0),
   green(0),
   blue(0)
{
   bconfig->addInt("red",   &red); 
   bconfig->addInt("green", &green); 
   bconfig->addInt("blue",  &blue); 

   bconfig->parse();
}


void DemoConfig::dump() const 
{
    bconfig->dump();
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    DemoConfig cfg("red=1,green=2,blue=3");
    cfg.dump();


    return 0 ; 
}

