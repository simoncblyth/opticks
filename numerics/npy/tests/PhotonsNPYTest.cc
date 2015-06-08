#include "NPY.hpp"
#include "PhotonsNPY.hpp"

int main(int argc, char** argv)
{
    PhotonsNPY oxc(NPY<float>::load("oxcerenkov", "1"));
    oxc.dump("oxc.dump");
    oxc.classify();

    oxc.classify(true);

    return 0 ;
}
