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

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "SSys.hh"
#include "BOpticksResource.hh"

#include "NPY.hpp"
#include "HitsNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "OPTICKS_LOG.hh"



struct HitsNPYTest
{
    HitsNPYTest( const char* idpath )
        :
        _res()
    {
        
        _res.setupViaID(idpath); 

        const char* idmpath = _res.getIdMapPath(); 
        assert( idmpath ); 

        _sens.load(idmpath);

    }

    void loadPhotons(const char* tag)
    {
        NPY<float>* photons = NPY<float>::load("oxtorch", tag,"dayabay");

        if(!photons)
        {
            LOG(error) << "failed to load photons " ;
            return  ;
        }    

        HitsNPY hits(photons, &_sens );
        hits.debugdump() ; 
    }

    bool             _testgeo ; 
    BOpticksResource _res ; 
    NSensorList      _sens ; 
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* idpath = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ; 

    HitsNPYTest hnt(idpath);
    hnt.loadPhotons("1");


    return 0 ;
}


