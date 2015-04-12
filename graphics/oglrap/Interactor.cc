#include "Interactor.hh"
#include "stdio.h"

#include <GLFW/glfw3.h>   // for the key definitions maybe recode to avoid this include 

#include "Camera.hh"
#include "View.hh"
#include "Trackball.hh"


const char* Interactor::DRAGFACTOR = "dragfactor" ; 
const char* Interactor::OPTIXMODE  = "optixmode" ; 

Interactor::Interactor() 
   :
   m_camera(NULL),
   m_view(NULL),
   m_trackball(NULL),
   m_zoom_mode(false), 
   m_pan_mode(false), 
   m_near_mode(false), 
   m_far_mode(false), 
   m_yfov_mode(false),
   m_rotate_mode(false),
   m_optix_mode(0),
   m_dragfactor(1.f)
{
}

void Interactor::configureF(const char* name, std::vector<float> values)
{
    printf("Interactor::configureF");
}

void Interactor::configureI(const char* name, std::vector<int> values)
{
    printf("Interactor::configureI");
    if(values.empty()) return ;
    if(strcmp(name, OPTIXMODE) == 0)
    {
        int last = values.back();
        setOptiXMode(last);
    }
}


void Interactor::setup(Camera* camera, View* view, Trackball* trackball)
{
    m_camera = camera ; 
    m_view   = view ; 
    m_trackball = trackball ; 
}   


void Interactor::cursor_drag(float x, float y, float dx, float dy )
{
    //printf("Interactor::cursor_drag x,y  %0.5f,%0.5f dx,dy  %0.5f,%0.5f \n", x,y,dx,dy );

    float df = m_dragfactor ; 

    if( m_yfov_mode )
    {
        m_camera->yfov_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_near_mode )
    {
        m_camera->near_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_far_mode )
    {
        m_camera->far_to(df*x,df*y,df*dx,df*dy);
    }
    else if( m_pan_mode )
    { 
        m_trackball->pan_to(df*x,df*y,df*dx,df*dy);
    } 
    else if( m_zoom_mode )
    { 
        m_trackball->zoom_to(df*x,df*y,df*dx,df*dy);
    } 
    else if( m_rotate_mode )
    {
        m_trackball->drag_to(df*x,df*y,df*dx,df*dy);
    }
    else
    {
    }
}

void Interactor::key_pressed(unsigned int key)
{
    switch (key)
    {
        case GLFW_KEY_Z:
            m_zoom_mode = !m_zoom_mode ; 
            break;
        case GLFW_KEY_X:
            m_pan_mode = !m_pan_mode ; 
            break;
        case GLFW_KEY_N:
            m_near_mode = !m_near_mode ; 
            break;
        case GLFW_KEY_F:
            m_far_mode = !m_far_mode ; 
            break;
        case GLFW_KEY_Y:
            m_yfov_mode = !m_yfov_mode ; 
            break;
        case GLFW_KEY_R:
            m_rotate_mode = !m_rotate_mode ; 
            break;
        case GLFW_KEY_O:
            m_optix_mode = !m_optix_mode ; 
            break;
        case GLFW_KEY_UP:
            m_dragfactor *= 2. ; 
            break;
        case GLFW_KEY_DOWN:
            m_dragfactor *= 0.5 ; 
            break;
        case GLFW_KEY_H:
            m_trackball->home(); 
            break;
 

    } 
    Print("Interactor::key_pressed");
}

void Interactor::key_released(unsigned int key)
{
   /*
    switch (key)
    {
        case GLFW_KEY_Z:
            m_zoom_mode = false ; 
            break;
        case GLFW_KEY_X:
            m_pan_mode = false ; 
            break;
        case GLFW_KEY_N:
            m_near_mode = false ; 
            break;
        case GLFW_KEY_F:
            m_far_mode = false ; 
            break;
        case GLFW_KEY_Y:
            m_yfov_mode = false ; 
            break;
    } 
    Print("Interactor::key_released");
   */
}


void Interactor::Print(const char* msg)
{
    printf("%s %s%s%s%s%s%s%s %10.3f \n", msg,
           m_zoom_mode ? "z" : "-",
           m_pan_mode  ? "x" : "-",
           m_far_mode  ? "f" : "-",
           m_near_mode ? "n" : "-",
           m_yfov_mode ? "y" : "-",
           m_rotate_mode ? "r" : "-",
           m_optix_mode ? "o" : "-",
           m_dragfactor );
}




