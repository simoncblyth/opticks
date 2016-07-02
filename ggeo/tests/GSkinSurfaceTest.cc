
#include "GOpticalSurface.hh"
#include "GSkinSurface.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GOpticalSurface* ops = NULL ; 

    GSkinSurface* sks = new GSkinSurface("test",0, ops);
    sks->Summary();


    return 0 ;
}

