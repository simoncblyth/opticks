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

// TEST=NCSGSaveTest om-t

#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNodeSample.hpp"
#include "NNode.hpp"

#include "OPTICKS_LOG.hh"

void test_load()
{
    NCSG* csg = NCSG::Load("$TMP/tboolean-box--/1"); 
    if(!csg) return ; 

    NPY<float>* gt = csg->getGTransformBuffer(); 
    assert(gt);  
    LOG(info) << " gt " << gt->getShapeString() ; 
}

void test_load_save()
{
    NCSG* csg = NCSG::Load("$TMP/tboolean-box--/1"); 
    NPY<float>* gt = csg->getGTransformBuffer(); 
    assert(gt);  

    if(!csg) return ; 
    csg->savesrc("$TMP/tboolean-box--save/1") ; 

    // savesrc after Load is an easy test to pass, as have the src buffers already from the loadsrc
}

void test_adopt_save()
{
    const char* name = "Box3" ; 
    //const char* name = "DifferenceOfSpheres" ; 

    nnode* sample = NNodeSample::Sample(name); 
    NCSG* csg = NCSG::Adopt(sample);
    assert( csg ); 

    const char* path = BStr::concat("$TMP/NCSGSaveTest/test_adopt_save/", name, NULL) ; 
    csg->savesrc(path); 


    // savesrc after Adopt is more difficult, depending on export_srcnode operation
}


const char* get_path(const char* prefix, const char* name, unsigned i )
{
    std::string path_ = BFile::FormPath(prefix, name, BStr::utoa(i) ) ; 
    const char* path = path_.c_str(); 
    return strdup(path); 
}

void test_chain()
{
    const char* name = "Box3" ; 
    nnode* sample = NNodeSample::Sample(name);
    NCSG* csg0 = NCSG::Adopt(sample);
    assert( csg0 ); 

    NCSG* csg = csg0 ; 

    const char* prefix = "$TMP/NCSGSaveTest/test_chain" ; 

    for(unsigned i=0 ; i < 5 ; i++)
    {
        const char* path = get_path( prefix, name, i ); 
        csg->savesrc(path);
        csg = NCSG::Load(path) ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_load(); 
    //test_load_save(); 
    //test_adopt_save(); 
    //test_chain();
 
    return 0 ; 
}


