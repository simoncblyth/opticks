#include "Lookup.hpp"

// canonical use in GGeo::setupLookup

int main(int, char**)
{
    Lookup::mockup("$TMP", "mockA.json", "mockB.json");

    Lookup lookup;
    lookup.loadA("$TMP", "mockA.json");
    lookup.loadB("$TMP", "mockB.json");
    lookup.crossReference();
    lookup.dump("LookupTest");



/* 
#include "G4StepNPY.hpp"
    const char* det = "dayabay" ; 
    G4StepNPY cs(NPY<float>::load("cerenkov", "1", det));
    cs.setLookup(&lookup);
    cs.applyLookup(0, 2); // materialIndex  (1st quad, 3rd number)
    cs.dump("cs.dump");
    cs.dumpLines("");

*/

    return 0 ;
}



