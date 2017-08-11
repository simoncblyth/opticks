
#include <cstdlib>
#include "NGLMExt.hpp"

#include "NHyperboloid.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"

void test_part()
{
    LOG(info) << "test_part" ; 
    nhyperboloid n = make_hyperboloid(100,100,-100,100);
    npart p = n.part();
    p.dump("hyp");
}

void test_bbox()
{
    LOG(info) << "test_bbox" ; 
    nhyperboloid n = make_hyperboloid(100,100,-100,100);
    nbbox bb = n.bbox();
    bb.dump("hyp");
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_part();
    test_bbox();

    return 0 ; 
}




