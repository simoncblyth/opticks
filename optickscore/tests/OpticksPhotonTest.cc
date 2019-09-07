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


#include "OpticksFlags.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
        
    LOG(info) << argv[0] ;

    for(unsigned i=0 ; i < 16 ; i++)
    {
        unsigned msk = 0x1 << i ;
        std::cout  
                  << " ( 0x1 << " << std::setw(2) << i << " ) "  
                  << " (i+1) " << std::setw(2) << std::hex << (i + 1) << std::dec
                  << " " << std::setw(2)  << OpticksFlags::FlagMask(msk, true) 
                  << " " << std::setw(20) << OpticksFlags::FlagMask(msk, false)
                  << " " << std::setw(6) << std::hex << msk << std::dec 
                  << " " << std::setw(6) << std::dec << msk << std::dec 
                  << std::endl 
                  ; 
 
    }

    return 0 ; 
}
