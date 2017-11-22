// ggv --opticks 

#include <iostream>

#include "BFile.hh"
#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"

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


void test_getDAEPath(Opticks* opticks)
{
    assert(opticks);
    const char* path = opticks->getDAEPath();
    test_path("getDAEPath", path);
}

void test_getGDMLPath(Opticks* opticks)
{
    assert(opticks);
    const char* path = opticks->getGDMLPath();
    test_path("getGDMLPath", path);
}


void test_getMaterialMap(Opticks* ok)
{
    assert(ok);
    const char* path = ok->getMaterialMap();
    test_path("getMaterialMap", path);
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


int main(int argc, char** argv)
{
    PLOG_(argc,argv);

    SYSRAP_LOG__ ; 
    BRAP_LOG__ ; 
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 
     
    LOG(info) << argv[0] ;
    Opticks ok(argc, argv);
    ok.configure();

    ok.Summary();

    LOG(info) << "OpticksTest::main aft configure" ;

    /*
    test_MaterialSequence();  
    test_getDAEPath(&ok);  
    test_getGDMLPath(&ok);  
    test_getMaterialMap(&ok);  
    */

    test_getDbgSeqhisMap(&ok);


    return 0 ;
}
