#include "Times.hpp"

#include <vector>


int main()
{
    std::vector<Times*> vt ; 

    Times* ck = Times::load("$IDPATH/times", "cerenkov_1.ini");
    Times* sc = Times::load("$IDPATH/times", "scintillation_1.ini") ;
    Times* cks = ck->clone();
    cks->setScale( 2817543./612841. );   // scale up according to photon count 

    ck->setLabel("ck");
    cks->setLabel("cks");
    sc->setLabel("sc");

    vt.push_back(ck);
    vt.push_back(cks);
    vt.push_back(sc);

    Times::compare(vt);

    return 0 ; 
}
