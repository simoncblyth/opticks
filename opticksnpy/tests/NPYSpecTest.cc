#include "NPYSpec.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPYSpec* spec = new NPYSpec(0,4,4,0, NPYBase::FLOAT) ;
    spec->Summary();

}
