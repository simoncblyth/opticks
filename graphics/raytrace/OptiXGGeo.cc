#include "OptiXGGeo.hh"

#include <optixu/optixu_vector_types.h>
#include "GGeo.hh"

/*
   Aiming to replace OptiXAssimpGeometry with this
   based on intermediary GGeo geometry structure

*/


OptiXGGeo::~OptiXGGeo()
{
}

OptiXGGeo::OptiXGGeo(GGeo* gg)
           : 
           m_ggeo(gg),
           m_context(NULL),
           m_material(NULL)
{
}


