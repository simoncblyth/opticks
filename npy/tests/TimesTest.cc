#include "Times.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>

int main(int, char** argv)
{

    std::vector<Times*> vt ; 

    if(getenv("IDPATH")==NULL)
    {
       std::cout << argv[0]
                 << "missing envvar IDPATH"
                 << std::endl ; 
       return 0 ; 
    }

    Times* ck = Times::load("ck", "$IDPATH/times", "cerenkov_1.ini");
    Times* sc = Times::load("sc", "$IDPATH/times", "scintillation_1.ini") ;
    Times* cks = ck->clone("cks");
    cks->setScale( 2817543./612841. );   // scale up according to photon count 

    vt.push_back(ck);
    vt.push_back(cks);
    vt.push_back(sc);

    Times::compare(vt);

    return 0 ; 
}
