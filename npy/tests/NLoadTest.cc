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

// TEST=NLoadTest om-t

#include <cassert>

#include "NPY.hpp"
#include "NLoad.hpp"

#include "OPTICKS_LOG.hh"

void test_Gensteps()
{
    NPY<float>* gs_0 = NLoad::Gensteps("dayabay","cerenkov","1") ;
    NPY<float>* gs_1 = NLoad::Gensteps("juno",   "cerenkov","1") ;
    NPY<float>* gs_2 = NLoad::Gensteps("dayabay","scintillation","1") ;
    NPY<float>* gs_3 = NLoad::Gensteps("juno",   "scintillation","1") ;

    assert(gs_0);
    assert(gs_1);
    assert(gs_2);
    assert(gs_3);

    std::string p0 = gs_0->getMeta<std::string>("path", ""); 
    std::string p1 = gs_1->getMeta<std::string>("path", ""); 
    std::string p2 = gs_2->getMeta<std::string>("path", ""); 
    std::string p3 = gs_3->getMeta<std::string>("path", ""); 

    LOG(info) << " p0 " << p0 ;  
    LOG(info) << " p1 " << p1 ;  
    LOG(info) << " p2 " << p2 ;  
    LOG(info) << " p3 " << p3 ;  

    //gs_0->dump();
    //gs_1->dump();
    //gs_2->dump();
    //gs_3->dump();
}

void test_directory()
{
    LOG(info) ; 
    std::string tagdir = NLoad::directory("pfx","det", "typ", "tag", "anno" ); 
    LOG(info) << " NLoad::directory(\"pfx\", \"det\", \"typ\", \"tag\", \"anno\" ) " << tagdir ; 
}

void test_reldir()
{
    LOG(info) ; 
    std::string rdir = NLoad::reldir("pfx", "det", "typ", "tag" ); 
    LOG(info) << " NLoad::reldir(\"pfx\", \"det\", \"typ\", \"tag\" ) " << rdir ; 
}



int main(int argc, char** argv)
{
     OPTICKS_LOG(argc, argv);

     NPYBase::setGlobalVerbose(true);

     test_Gensteps();  
     test_directory();
     test_reldir();

 
     return 0 ; 
}
