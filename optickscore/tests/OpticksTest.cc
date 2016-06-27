// ggv --opticks 

#include <iostream>

#include "BFile.hh"
#include "PLOG.hh"

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






int main(int argc, char** argv)
{
    PLOG_(argc,argv);
    LOG(info) << argv[0] ;

    //plog::Logger<0>* logger = plog::get(); 

    BRAP_LOG__ ; 
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 
     

    Opticks ok(argc, argv);

    //ok.Summary();


    ok.configure();
    LOG(info) << "OpticksTest::main aft configure" ;
 
    test_MaterialSequence();  
    test_getDAEPath(&ok);  
    test_getGDMLPath(&ok);  



    return 0 ;
}
