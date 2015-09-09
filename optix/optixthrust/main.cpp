#include "optixthrust.hh"

int main( int argc, char** argv )
{
    OptiXThrust ot(100);

    ot.photon_test();

    ot.compile();

    //ot.minimal();
    ot.circle();
    ot.dump();

    ot.compaction();

    //ot.postprocess();


    return 0 ; 
} 
