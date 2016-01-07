#include "Opticks.hh"

#include "GCache.hh"
#include "GSource.hh"


int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "source.log");

    GCache* cache = new GCache(opticks);

    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );

    source->Summary();


    return 0 ; 
}

