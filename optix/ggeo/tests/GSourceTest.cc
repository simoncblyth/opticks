#include "GCache.hh"
#include "GSource.hh"


int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "source.log", "info");

    cache->configure(argc, argv);

    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );

    source->Summary();

    


    return 0 ; 
}

