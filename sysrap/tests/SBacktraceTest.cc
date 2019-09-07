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


#include "SBacktrace.hh"
#include "OPTICKS_LOG.hh"


struct SBT
{
   static void red(); 
   static void green(); 
   static void blue(); 
   static void cyan(); 
   static void magenta(); 
   static void yellow(); 

};

void SBT::red(){  
   LOG(info) << "." ; 
   SBacktrace::Dump(); 
}
void SBT::green(){  
   LOG(info) << "." ; 
   red(); 
}
void SBT::blue(){  
   LOG(info) << "." ; 
   green(); 
}
void SBT::cyan(){  
   LOG(info) << "." ; 
   blue(); 
}
void SBT::magenta(){  
   LOG(info) << "." ; 
   cyan(); 
}
void SBT::yellow(){  
   LOG(info) << "." ; 
   magenta(); 
}

int main(int argc, char** argv)
{  
    OPTICKS_LOG(argc, argv); 
    SBT::yellow();      
    return 0 ; 
}
