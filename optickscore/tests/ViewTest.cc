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

#include <iomanip>
#include <cassert>


#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "View.hh"


#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    View v ; 
    //v.setEye(1,0,0) ;

    v.Summary();

    //glm::mat4 m2w ; 

    
    
    return 0 ;
}

/*

View::Summary
            eye vec4      -1.000     -1.000      0.000      1.000 
           look vec4       0.000      0.000      0.000      1.000 
             up vec4       0.000      0.000      1.000      0.000 


*/


