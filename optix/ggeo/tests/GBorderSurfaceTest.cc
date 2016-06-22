
#include "GBorderSurface.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GBorderSurface* bs = new GBorderSurface("test", 0, NULL );
    bs->Summary();


    return 0 ;
}

