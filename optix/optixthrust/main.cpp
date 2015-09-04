#include "optixthrust.hh"

int main( int argc, char** argv )
{
    OptiXThrust ot;

    ot.compile();
    ot.launch(OptiXThrust::raygen_minimal_entry);
    ot.postprocess();
    ot.launch(OptiXThrust::raygen_dump_entry);   // OptiX succeeds to see the changes made by Thrust 

    return 0 ; 
} 
