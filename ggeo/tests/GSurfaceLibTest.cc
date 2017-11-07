// op --surf
// op --surf 6        // summary of all and detail of just the one index
// op --surf lvPmtHemiCathodeSensorSurface


#include "Opticks.hh"

#include "GAry.hh"
#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GSurfaceLib.hh"

#include "GGEO_BODY.hh"
#include "GGEO_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ; 
    NPY_LOG__ ; 

    Opticks* opticks = new Opticks(argc, argv);

    GSurfaceLib* m_slib = GSurfaceLib::load(opticks);

    m_slib->dump();


    // cf CPropLib::init
/*
    GPropertyMap<float>* m_sensor_surface = NULL ; 

    m_sensor_surface = m_slib->getSensorSurface(0) ;

    if(m_sensor_surface == NULL)
    {   
        LOG(warning) << "GSurfaceLibTest"
                     << " surface lib sensor_surface NULL "
                     ;   
    }   
    else
    {   
        m_sensor_surface->Summary("cathode_surface");
    }   

*/


    return 0 ;
}

