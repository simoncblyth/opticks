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

#include "Opticks.hh"
#include "OpticksAna.hh"

#include "OPTICKS_LOG.hh"


struct OpticksAnaTest 
{   
    OpticksAnaTest(const Opticks* ok)
    {
        OpticksAna* ana = ok->getAna(); 
        const char* anakey = ok->getAnaKey(); 
        std::string cmdline = ana->getCommandLine(anakey ? anakey : "tboolean");  
        LOG(info) 
            << "anakey " << anakey
            ;
        std::cout << cmdline << std::endl ; 

    }

};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ; 
    ok.configure();
    //ok.ana();

    OpticksAnaTest oat(&ok); 
 
    return ok.getRC();
}

/**

::

   OpticksAnaTest --anakey tpmt --tag 10 --cat PmtInBox


**/

