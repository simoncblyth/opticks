#include "NPY.hpp"
#include "PhotonsNPY.hpp"

int main(int argc, char** argv)
{
    PhotonsNPY oxc(NPY::load("oxcerenkov", "1"));
    oxc.dump("oxc.dump");
    oxc.classify();

    return 0 ;
}
