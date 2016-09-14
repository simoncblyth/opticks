#include "FabStepNPY.hpp"



FabStepNPY::FabStepNPY(unsigned genstep_type, unsigned num_step, unsigned num_photons_per_step) 
       :  
       GenstepNPY(genstep_type, num_step),
       m_num_photons_per_step(num_photons_per_step)
{
    init();
}

void FabStepNPY::init()
{
    unsigned num_step = getNumStep();
    for(unsigned i=0 ; i < num_step ; i++)
    {   
        setMaterialLine(i*10);   
        setNumPhotons(m_num_photons_per_step); 
        addStep();
    } 
}

void FabStepNPY::update()
{
}
