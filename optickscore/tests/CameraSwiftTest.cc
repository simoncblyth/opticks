#include "NGLM.hpp"
#include "GLMPrint.hpp"

#include "Camera.hh"

int main()
{
    float basis = 100.f ; 
    Camera* c = new Camera(1024, 768, basis)  ;
    c->Summary();

}
