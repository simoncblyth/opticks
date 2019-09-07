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

#include "STranche.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

   
    STranche st0( 1000, 100 );
    st0.dump();
 
    STranche st1( 1013, 100 );
    st1.dump();

    STranche st2( 1099, 100 );
    st2.dump();

    STranche st3( 1100, 100 );
    st3.dump();

    STranche st4( 1101, 100 );
    st4.dump();



    return 0  ; 
}
