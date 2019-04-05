
#include <cstdlib>
#include "NGLMExt.hpp"

#include "NCubic.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"

void test_part()
{
    LOG(info) << "test_part" ; 
    ncubic* n = make_cubic(1,1,1,1,-1,1);
    npart p = n->part();
    p.dump("cubic");
}

void test_bbox()
{
    LOG(info) << "test_bbox" ; 
    ncubic* n = make_cubic(1,1,1,1,-1,1);
    nbbox bb = n->bbox();
    bb.dump("hyp");
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_part();
    test_bbox();

    return 0 ; 
}




