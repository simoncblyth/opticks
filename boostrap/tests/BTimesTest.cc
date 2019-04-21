#include "BTimes.hh"

#include <iostream>
#include <vector>
#include <cstdlib>

int main(int, char** argv)
{

    std::vector<BTimes*> vt ; 

    if(getenv("IDPATH")==NULL)
    {
       std::cout << argv[0]
                 << "missing envvar IDPATH"
                 << std::endl ; 
       return 0 ; 
    }

    BTimes* ck = BTimes::load("ck", "$IDPATH/times", "cerenkov_1.ini");
    BTimes* sc = BTimes::load("sc", "$IDPATH/times", "scintillation_1.ini") ;
    BTimes* cks = ck->clone("cks");
    cks->setScale( 2817543./612841. );   // scale up according to photon count 

    vt.push_back(ck);
    vt.push_back(cks);
    vt.push_back(sc);

    BTimes::compare(vt);

    return 0 ; 
}
