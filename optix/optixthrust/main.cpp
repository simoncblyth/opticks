#include "optixthrust.hh"

int main( int argc, char** argv )
{
    OptiXThrust ot(100);

    ot.photon_test();

    ot.compile();

    //ot.minimal();
    ot.circle();
    ot.dump();

    //ot.compaction();
    //ot.strided();
    //ot.strided4();
    ot.compaction4();

    //ot.postprocess();

    ot.sync();


    return 0 ; 
} 
