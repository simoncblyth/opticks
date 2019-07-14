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
        const char* path = ok->getDirectGenstepPath();
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





void test_getDAEPath(Opticks* ok)
{
    assert(ok);
    const char* path = ok->getDAEPath();
    test_path("getDAEPath", path);
}

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





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);
    
    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    ok.configure();

    ok.Summary();

    LOG(info) << "OpticksTest::main aft configure" ;

    /*
    test_MaterialSequence();  
    test_getDAEPath(&ok);  
    test_getMaterialMap(&ok);  
    test_getDbgSeqhisMap(&ok);
    test_gpumon(&ok);
    test_getGDMLPath(&ok);  
    test_loadCacheMeta(&ok);  
    */

    test_getCurrentGDMLPath(&ok); 

    //OpticksTest okt(&ok); 
    //okt.test_getGenstepPath();  
    //okt.test_getDirectGenstepPath();  


    return 0 ;
}



