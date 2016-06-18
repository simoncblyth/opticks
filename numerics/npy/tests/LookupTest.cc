#include "G4StepNPY.hpp"
#include "Lookup.hpp"


int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("%s : expecting argument directory containing %s \n", argv[0], Lookup::BNAME);
        return 1 ;
    }

    Lookup lookup;
    lookup.create(argv[1]);
    lookup.dump("LookupTest");
 
    const char* det = "dayabay" ; 

    G4StepNPY cs(NPY<float>::load("cerenkov", "1", det));
    

    cs.setLookup(&lookup);
    cs.applyLookup(0, 2); // materialIndex  (1st quad, 3rd number)

    cs.dump("cs.dump");
    cs.dumpLines("");

    return 0 ;
}



