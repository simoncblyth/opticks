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
