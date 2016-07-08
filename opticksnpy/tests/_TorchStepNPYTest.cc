#include "TorchStepNPY.hpp"
#include "NPY.hpp"

#ifdef _MSC_VER
// quell: object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif



int main(int , char** )
{
    TorchStepNPY* m_torchstep ; 

    //const char* config = "target=3153;photons=10000;dir=0,1,0" ;
    const char* config = NULL ;

    m_torchstep = new TorchStepNPY(0, 1, config);

    m_torchstep->dump();

    m_torchstep->addStep();

    NPY<float>* npy = m_torchstep->getNPY();
    npy->save("$TMP/torchstep.npy");


    return 0 ;
}; 
