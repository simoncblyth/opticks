

#include <iostream>
#include <iomanip>
#include <vector>

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "Composition.hh"
#include "FlightPath.hh"
#include "View.hh"
#include "Animator.hh"
#include "InterpolatedView.hh"

#include "NP.hh"


struct MockRenderer
{
    static const char* FILENAME ; 
    static int Preinit(); 
    MockRenderer(Opticks* ok); 
    void init(); 
    void renderLoop();
    void render(unsigned count); 
    void render_debug(unsigned count); 
    void save(const char* dir);

    int          m_preinit ; 
    Opticks*     m_ok ;
    int          m_limit ;  
    Composition* m_composition ; 
    FlightPath*  m_flightpath ; 

    std::vector<glm::vec4> m_elui ; 

}; 


int MockRenderer::Preinit()
{
    LOG(info); 
    return 0 ; 
}


MockRenderer::MockRenderer(Opticks* ok)
    :
    m_preinit(Preinit()),
    m_ok(ok), 
    m_limit(0),
    m_composition(new Composition(m_ok)),
    m_flightpath(new FlightPath(m_ok->getFlightPathDir()))
{
    init(); 
}


void  MockRenderer::init()
{
    LOG(info) << "[" ; 

    
    m_composition->setFlightPath(m_flightpath); 
    m_composition->setViewType(View::FLIGHTPATH); 


    View* view = m_composition->getView(); 

    InterpolatedView* iv = reinterpret_cast<InterpolatedView*>(view); 
    assert(iv); 
    iv->commandMode("TB") ;  // FAST16
    //iv->commandMode("TC") ;  // FAST32
    // iv->commandMode("TD") ;  // FAST64  loadsa nan

    unsigned num_views = iv->getNumViews();  
    Animator* anim = iv->getAnimator(); 
    unsigned period = anim->getPeriod();  

    unsigned tot_period = period*num_views ;

    m_limit = tot_period ; 

    LOG(info) 
        << " num_views " << num_views
        << " animator.period " << period
        << " tot_period " << tot_period
        << " iv " << iv
        ;


    LOG(info) << "]" ; 
}


// mocking OpticksViz::renderLoop
void MockRenderer::renderLoop()
{
    LOG(info) << "[" ; 

    unsigned count(0) ; 
    bool exitloop(false); 

    int renderlooplimit = m_limit > 0 ? m_limit : m_ok->getRenderLoopLimit(); 

    while (!exitloop  )
    {   
        count = m_composition->tick();

        //if( m_composition->hasChanged() || count == 1)  
        //{   
            render(count);

           // m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        //}   
        exitloop = renderlooplimit > 0 && int(count) > renderlooplimit ; 
    }   
    LOG(info) << "]" ; 
}


void MockRenderer::render(unsigned count)
{
    // OTracer::trace_() does the below : so that is what matters
    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    glm::vec4 ZProj ;

    m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first

    std::cout 
        << std::setw(5) << count 
        << " eye ("
        << " " << std::setw(10) << eye.x
        << " " << std::setw(10) << eye.y
        << " " << std::setw(10) << eye.z
        << ")"
        << " U ("
        << " " << std::setw(10) << U.x
        << " " << std::setw(10) << U.y
        << " " << std::setw(10) << U.z
        << ")"
        << " V ("
        << " " << std::setw(10) << V.x
        << " " << std::setw(10) << V.y
        << " " << std::setw(10) << V.z
        << ")"
        << " W ("
        << " " << std::setw(10) << W.x
        << " " << std::setw(10) << W.y
        << " " << std::setw(10) << W.z
        << ")"
        << std::endl 
        ;     
  
}



void MockRenderer::render_debug(unsigned count)
{

    View* view = m_composition->getView(); 
    assert( view->isInterpolated() ); 
    InterpolatedView* iv = reinterpret_cast<InterpolatedView*>(m_composition->getView()); 
    assert(iv);

    glm::mat4 identity(1.f); 
    glm::vec4 ive = iv->getEye(identity); 
    glm::vec4 ivl = iv->getLook(identity); 
    glm::vec4 ivu = iv->getUp(identity); 
    glm::vec4 spare(0.f, 0.f, 0.f, 0.f); 


    m_elui.push_back(ive); 
    m_elui.push_back(ivl); 
    m_elui.push_back(ivu); 
    m_elui.push_back(spare); 


    std::cout 
        << std::setw(5) << count 
        << " view.typename " << view->getTypeName()
/*
   // these dont budge
        << " composition.getEye (" 
        << " " << std::setw(10) << m_composition->getEyeX() 
        << " " << std::setw(10) << m_composition->getEyeY() 
        << " " << std::setw(10) << m_composition->getEyeZ() 
        << ")"
        << " view.getEyeX,Y,Z (" 
        << " " << std::setw(10) << view->getEyeX() 
        << " " << std::setw(10) << view->getEyeY() 
        << " " << std::setw(10) << view->getEyeZ() 
        << ")"
        << " iv.getEyeX,Y,Z (" 
        << " " << std::setw(10) << iv->getEyeX() 
        << " " << std::setw(10) << iv->getEyeY() 
        << " " << std::setw(10) << iv->getEyeZ() 
        << ")"
*/
        << " ive (" 
        << " " << std::setw(10) << ive.x 
        << " " << std::setw(10) << ive.y
        << " " << std::setw(10) << ive.z 
        << " " << std::setw(10) << ive.w 
        << ")"
        << std::endl 
        ;    

    std::cout 
        << std::setw(5) << count 
        << " " 
        << m_composition->getEyeString() 
        << std::endl
        ; 

}

const char* MockRenderer::FILENAME = "FlightPathTest.npy" ; 

void MockRenderer::save(const char* dir)
{
    NP::Write(dir, FILENAME, (float*)m_elui.data(), m_elui.size(), 4 ) ; 
}


void test_MockRenderer(Opticks* ok)
{
    MockRenderer renderer(ok); 
    renderer.renderLoop();  
    renderer.save("/tmp"); 
}

void test_fillPathFormat(Opticks* ok)
{
    LOG(info); 

    FlightPath* fp = ok->getFlightPath(); 

    fp->setPathFormat("$TMP", "FlightPathTest");  

    char path[128]; 
    for(unsigned index=0 ; index < 10 ; index++)
    {
        fp->fillPathFormat(path, 128, index ); 
        std::cout 
            << std::setw(4) << index 
            << " : "
            << path 
            << std::endl 
            ;
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    Opticks* ok = new Opticks(argc, argv); 
    ok->configure(); 

    //test_MockRenderer(ok) ;
    
    test_fillPathFormat(ok); 

    return 0 ;
}

