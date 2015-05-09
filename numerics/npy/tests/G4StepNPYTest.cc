
#include "NPY.hpp"
#include "G4StepNPY.hpp"

int main()
{
    NPY* npy = NPY::load("cerenkov", "1");

    G4StepNPY cs(npy);
    cs.loadChromaMaterialMap("/tmp/ChromaMaterialMap.json");
    cs.dumpChromaMaterialMap();


    return 0 ;
}
