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



#include "NPY.hpp"
#include "GLMPrint.hpp"



#include "Opticks.hh"
#include "OpticksAttrSeq.hh"

#include "GBndLib.hh"
#include "BoundariesNPY.hpp"

#include "OPTICKS_LOG.hh"

#include "GGEO_BODY.hh"


// ggv --boundaries

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv);
    ok->configure();

    GBndLib* blib = GBndLib::load(ok, true );
    OpticksAttrSeq* qbnd = blib->getAttrNames();
    blib->close();     //  BndLib is dynamic so requires a close before setNames is called setting the sequence for OpticksAttrSeq
    std::map<unsigned int, std::string> nm = qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;

    qbnd->dump();
    
    NPY<float>* dpho = NPY<float>::load("oxtorch", "1", "dayabay");
    if(!dpho) 
    {
        LOG(warning) << " failed to load dpho event " ; 
        return 0 ;
    }


    //dpho->Summary();

    BoundariesNPY* boundaries = new BoundariesNPY(dpho);
    boundaries->setBoundaryNames(nm); 
    boundaries->setTypes(NULL);
    boundaries->indexBoundaries();
    boundaries->dump();

    //glm::ivec4 sel = boundaries->getSelection() ;
    //print(sel, "boundaries selection");

    return 0 ; 

}
