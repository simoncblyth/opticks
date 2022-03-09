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

// op --opticks    ## handy to be within envvar environ in legacy mode
// TEST=OpticksTest om-t 

#include <iostream>

#include "BFile.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"


void test_MaterialSequence()
{

    unsigned long long seqmat = 0x0123456789abcdef ;

    std::string s_seqmat = Opticks::MaterialSequence(seqmat) ;

    LOG(info) 
              << "OpticksTest::main"
              << " seqmat "
              << std::hex << seqmat << std::dec
              << " MaterialSequence " 
              << s_seqmat
              ;
}



void test_path(const char* msg, const char* path)
{
    std::string npath = BFile::FormPath(path);
    bool exists = BFile::ExistsFile(path);
    LOG(info) 
              << msg  
              << " path " << path 
              << " npath " << npath 
              << " exists " << exists 
              ;
}


struct OpticksTest 
{
    OpticksTest(const Opticks* ok_) : ok(ok_) 
    {
        assert(ok);
    }  

    void test_getDirectGenstepPath()
    {
        const char* path = ok->getDirectGenstepPath(0);
        test_path("getDirectGenstepPath", path );
    }

/*
    void test_getGenstepPath()
    {
        const char* path = ok->getGenstepPath();
        test_path("getGenstepPath", path );
    }
*/




    const Opticks* ok ; 

};




#ifdef OLD_RESOURCE
void test_getDAEPath(Opticks* ok)
{
    assert(ok);
    const char* path = ok->getDAEPath();
    test_path("getDAEPath", path);
}

void test_getMaterialMap(Opticks* ok)
{
    assert(ok);
    const char* path = ok->getMaterialMap();
    test_path("getMaterialMap", path);
}

#endif


void test_getGDMLPath(Opticks* ok)
{
    assert(ok);
    const char* path0 = ok->getSrcGDMLPath();
    test_path("getSrcGDMLPath", path0);

    const char* path1 = ok->getGDMLPath();
    test_path("getGDMLPath", path1);

    const char* path2 = ok->getCurrentGDMLPath();
    test_path("getCurrentGDMLPath", path2);
}



void test_getDbgSeqhisMap(Opticks* ok)
{
     unsigned long long seqhis(0) ;
     unsigned long long seqval(0) ;

     const std::string& seqmap = ok->getSeqMapString();
     bool has_seqmap = ok->getSeqMap(seqhis, seqval);

     LOG(info) 
           << " seqmap " << seqmap
           << " has: " << ( has_seqmap ? "Y" : "N" )
           ;

     if(has_seqmap)
     {
         LOG(info)
               << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec 
               << " seqval " << std::setw(16) << std::hex << seqval << std::dec 
               ;
     }
}
/*
    OpticksTest --seqmap "TO:0 SR:1 SA:0" 
    OpticksTest --seqmap "TO:0,SR:1,SA:0" 
    OpticksTest --seqmap "TO:0,SR:1,SA:0" 
    OpticksTest --seqmap "TO:0 SR:1 SA:0" 
    OpticksTest --seqmap "TO:0 SC: SR:1 SA:0" 
*/


void test_gpumon(Opticks* ok)
{
   LOG(info)
        << " --gpumonpath " << ok->getGPUMonPath()
        << " --gpumon " << ok->isGPUMon()
        ;
         
}


void test_getCurrentGDMLPath(Opticks* ok)
{
    const char* gdmlpath = ok->getCurrentGDMLPath();
    LOG(info) << gdmlpath ;  
}


void test_getEventFold(Opticks* ok)
{
    const char* ef = ok->getEventFold();
    LOG(info) << ef ;  
}

void test_findGDMLAuxMetaEntries(const Opticks* ok)
{
    const char* key = "label" ; 
    const char* val = "target" ; 
    //const char* val = NULL ; 

    std::vector<BMeta*> entries ; 
    ok->findGDMLAuxMetaEntries(entries, key, val ); 
}

void test_findGDMLAuxValues( const Opticks* ok)
{
    const char* k = "label" ; 
    const char* v = "target" ; 
    const char* q = "lvname" ; 

    std::vector<std::string> values ; 
    ok->findGDMLAuxValues(values, k,v,q); 

    LOG(info) 
        << " for entries matching (k,v) : " << "(" << k << "," << v << ")" 
        << " collect values of q:" << q
        << " : values.size() " << values.size()
        ;

    for(unsigned i=0 ; i < values.size() ; i++) std::cout << values[i] << std::endl ; 
}

void test_getGDMLAuxTargetLVName(const Opticks* ok)
{
    const char* lvn = ok->getGDMLAuxTargetLVName(); 
    LOG(info) << lvn ; 
}

void test_OriginGDMLPath()
{
    LOG(info) << Opticks::OriginGDMLPath() ; 
}


void test_isEnabledMergedMesh(const Opticks* ok)
{
    LOG(info) << " ok.emm " << ok->getEnabledMergedMesh() ; 
    for(unsigned i=0 ; i < 16 ; i++)
    {
        std::cout 
            << std::setw(4) << i  
            << std::setw(4) << ok->isEnabledMergedMesh(i) 
            << std::endl
            ;  
    }
}

void test_getFlightInputDir(const Opticks* ok)
{
    std::string dir = ok->getFlightInputDir(); 
    LOG(info) << dir ; 
    std::string path = ok->getFlightInputPath(); 
    LOG(info) << path  ; 
}



void test_getArgList(const Opticks* ok)
{
    const std::vector<std::string>& arglist = ok->getArgList();   // --arglist /path/to/arglist.txt

    LOG(info) << " arglist.size " << arglist.size(); 
    for(unsigned i=0 ; i < arglist.size() ; i++) LOG(info) << "[" << arglist[i] << "]" ; 
}

void test_getIdPath(const Opticks* ok)
{
    const char* idpath = ok->getIdPath(); 
    const char* geocachedir = ok->getGeocacheDir(); 
    LOG(info) << " idpath      " << idpath ; 
    LOG(info) << " geocachedir " << geocachedir ; 
}

void test_writeGeocacheScript(const Opticks* ok)
{
    ok->writeGeocacheScript(); 
}

void test_isGPartsTransformOffset(const Opticks* ok)
{
    bool is_gparts_transform_offset = ok->isGPartsTransformOffset(); 
    LOG(info) << " is_gparts_transform_offset " << is_gparts_transform_offset ; 
}


void test_getCacheMetaTime(const Opticks* ok)
{
    const char* path = ok->getCacheMetaPath(); 
    int mtime = ok->getCacheMetaTime(); 
    std::string stamp = ok->getCacheMetaStamp(); 
    LOG(info) 
       << " path " << path 
       << " mtime " << mtime 
       << " stamp " << stamp
       ; 
}

void test_isCXSkipLV(const Opticks* ok)
{
    unsigned numCXSkipLV = ok->getNumCXSkipLV(); 
    LOG(info) << "numCXSkipLV " << numCXSkipLV << " eg --cxskiplv 1,2,3,101 " ; 

    for(unsigned i=0 ; i < 1000 ; i++ )
    {
        unsigned meshIdx = i ; 
        bool cxSkip = ok->isCXSkipLV(meshIdx) ; 
        if(cxSkip) LOG(info) << " --cxskiplv meshIdx " << meshIdx ; 
    }
}

void test_getSize(const Opticks* ok)
{
    glm::uvec4 sz = ok->getSize(); 
    LOG(info) << " sz : " << glm::to_string(sz ) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);
    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    ok.configure();

    /*
    ok.Summary();

    test_MaterialSequence();  
    test_getDAEPath(&ok);  
    test_getMaterialMap(&ok);  
    test_getDbgSeqhisMap(&ok);
    test_gpumon(&ok);
    test_getGDMLPath(&ok);  
    test_loadCacheMeta(&ok);  
    test_getCurrentGDMLPath(&ok); 

    OpticksTest okt(&ok); 
    okt.test_getGenstepPath();  
    okt.test_getDirectGenstepPath();  

    test_getEventFold(&ok); 
    test_findGDMLAuxMetaEntries(&ok); 
    test_findGDMLAuxValues(&ok); 
    test_getGDMLAuxTargetLVName(&ok); 

    test_OriginGDMLPath(); 
    test_isEnabledMergedMesh(&ok); 
    test_getFlightInputDir(&ok); 

    test_getArgList(&ok); 
    test_getIdPath(&ok); 
    test_writeGeocacheScript(&ok); 
    test_isGPartsTransformOffset(&ok); 
    test_getCacheMetaTime(&ok); 
    test_isCXSkipLV(&ok); 

    */

    test_getSize(&ok); 

    return 0 ;
}



