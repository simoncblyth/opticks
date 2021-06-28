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

// op --surf
// op --surf 6        // summary of all and detail of just the one index
// op --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"

#include "GAry.hh"
#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GSurfaceLib.hh"

#include "GGEO_BODY.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv);
    ok->configure();

    GSurfaceLib* m_slib = GSurfaceLib::load(ok);

    m_slib->dump();


    return 0 ;
}

