#include "GSource.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;  


    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );

    source->Summary();


    return 0 ; 
}

