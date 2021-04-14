

#include <iostream>
#include <iomanip>

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "Composition.hh"
#include "FlightPath.hh"
#include "View.hh"

struct MockRenderer
{
    MockRenderer(Opticks* ok); 

    void renderLoop();
    void render(unsigned count); 

    Opticks*     m_ok ; 
    Composition* m_composition ; 
    FlightPath*  m_flightpath ; 
}; 


MockRenderer::MockRenderer(Opticks* ok)
    :
    m_ok(ok), 
    m_composition(new Composition(m_ok)),
    m_flightpath(new FlightPath(m_ok->getFlightPathDir()))
{
    m_composition->setFlightPath(m_flightpath); 

    m_composition->setViewType(View::FLIGHTPATH); 
}

// mocking OpticksViz::renderLoop
void MockRenderer::renderLoop()
{
    unsigned count(0) ; 
    bool exitloop(false); 

    int renderlooplimit = m_ok->getRenderLoopLimit(); 

    while (!exitloop  )
    {   
        count = m_composition->tick();

        if( m_composition->hasChanged() || count == 1)  
        {   
            render(count);

            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }   
        exitloop = renderlooplimit > 0 && int(count) > renderlooplimit ; 
    }   
}

void MockRenderer::render(unsigned count)
{
    std::cout 
        << std::setw(5) << count 
        << " eye (" 
        << " " << std::setw(10) << m_composition->getEyeX() 
        << " " << std::setw(10) << m_composition->getEyeY() 
        << " " << std::setw(10) << m_composition->getEyeZ() 
        << ")"
        << std::endl 
        ;    
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks* m_ok = new Opticks(argc, argv); 
    m_ok->configure(); 

    MockRenderer renderer(m_ok); 
    renderer.renderLoop();  

    return 0 ;
}

