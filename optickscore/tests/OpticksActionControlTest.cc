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

#include "OpticksActionControl.hh"
#include <iostream>
#include <iomanip>

#include "OPTICKS_LOG.hh"

typedef std::vector<const char*> VC ; 

void dump(const OpticksActionControl& ctrl, const char* msg="")
{
     LOG(info) << msg ; 

     VC tags = OpticksActionControl::Tags();

     for(VC::const_iterator it=tags.begin() ; it != tags.end() ; it++)
     {
         const char* tag = *it ; 
         bool set = ctrl.isSet(tag) ;

         LOG(info) << std::setw(20) << tag 
                   << " " << ( set ? "Y" : "N" )
                   ;
     } 
}


int main(int argc, char** argv)
{
     OPTICKS_LOG(argc, argv);

     const char* ctrl_ = "GS_LOADED,GS_FABRICATED,GS_TRANSLATED" ;
     unsigned long long mask = OpticksActionControl::Parse(ctrl_) ;

     std::cout << " ctrl " << ctrl_ 
               << " mask " << mask 
               << " desc " << OpticksActionControl::Description(mask)
               << std::endl ; 


     OpticksActionControl c0(&mask);
     dump(c0);

     const char* label = "GS_TORCH" ; 

     c0.add(label);
     assert(c0.isSet(label));

     dump(c0, "after addition");
      
 

     return 0 ; 
}
