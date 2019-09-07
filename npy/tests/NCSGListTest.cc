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

// TEST=NCSGListTest om-t

#include "OPTICKS_LOG.hh"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NPYBase.hpp"
#include "NPYList.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* csgpath = argc > 1 ? argv[1] : "$TMP/tboolean-box--" ; 
    unsigned verbosity = 0 ; 

    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 
   
    NCSGList* ls = NCSGList::Load(csgpath, verbosity );    
    if( ls == NULL )
    {
        LOG(warning) << "FAILED to load NCSG trees from " << csgpath  ; 
        return 0 ;
    }

    ls->dumpDesc();
    ls->dumpMeta();
    ls->dumpUniverse();

    unsigned num_trees = ls->getNumTrees(); 

    for(unsigned i=0 ; i < num_trees ; i++)
    {
        NCSG* tree = ls->getTree(i) ; 
        NPYList* npy = tree->getNPYList(); 
        LOG(info) << npy->desc() ; 

        NPY<float>* gt = tree->getGTransformBuffer(); 

        if(!gt) LOG(fatal) << "NO GTransformBuffer " ; 
        //assert(gt); 
    }




    return 0 ; 
}
