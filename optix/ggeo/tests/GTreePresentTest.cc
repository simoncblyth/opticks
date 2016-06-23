
#include <cassert>
#include "GTreePresent.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GTreePresent* tp = new GTreePresent(100, 1000) ;  // depth_max sibling_max
    tp->traverse(NULL);




    return 0 ;
}

