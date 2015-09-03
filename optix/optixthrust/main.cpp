#include "optixthrust.hh"

int main( int argc, char** argv )
{
    OptiXThrust ot;
    ot.launch();
    ot.postprocess();
    return 0 ; 
} 
