// TEST=BMetaTest om-t 

#include <iostream>
#include <iomanip>
#include <string>

#include "OPTICKS_LOG.hh"
#include "BMeta.hh"




struct BMetaTest
{
    BMetaTest()
    {
        const char* label = "some_label" ; 

        BMeta* a = new BMeta(label) ; 

        a->add("red","a"); 
        a->add("green","b"); 
        a->add("blue","c"); 
        a->addEnvvar("CUDA_VISIBLE_DEVICES");
        a->addEnvvar("OPTICKS_RTX");
        a->addEnvvar("OPTICKS_KEY");

        a->dump();

        const char* dir = "$TMP/boostrap/tests/BMetaTest" ; 
        a->save(dir); 


        BMeta* b = BMeta::Load( dir, label ); 
        b->dump(); 
    }
};



int main(int argc, char** argv, char** envp)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info); 

    BMetaTest mt ; 


    return 0 ; 
}
