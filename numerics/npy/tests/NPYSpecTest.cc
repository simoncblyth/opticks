#include "Blog.hh"
#include "NPYSpec.hpp"

int main(int argc, char** argv)
{
    BLOG(argc, argv);
    NPYSpec* spec = new NPYSpec(0,4,4,0, NPYBase::FLOAT) ;
    spec->Summary();

}
