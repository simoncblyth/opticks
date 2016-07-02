#include "GTransforms.hh"


int main()
{
    GTransforms* t = new GTransforms() ; 

    t->add();
    t->add();
    t->add();

    t->save("/tmp/transforms.npy");

    return 0 ; 
}
