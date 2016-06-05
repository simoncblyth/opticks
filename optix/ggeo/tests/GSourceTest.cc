#include "GSource.hh"


int main(int argc, char** argv)
{
    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );

    source->Summary();


    return 0 ; 
}

