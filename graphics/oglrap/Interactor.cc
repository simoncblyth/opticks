#include "Interactor.hh"
#include "stdio.h"

#include "Frame.hh"
//#include <GLFW/glfw3.h>   // for the key definitions maybe recode to avoid this include 

#include "Composition.hh"
#include "Bookmarks.hh"

#include "Camera.hh"
#include "View.hh"
#include "Trackball.hh"
#include "Clipper.hh"
#include "Touchable.hh"
#include "Scene.hh"


#include <string>
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#ifdef GUI_
#include <imgui.h>
#endif



const char* Interactor::DRAGFACTOR = "dragfactor" ; 
const char* Interactor::OPTIXMODE  = "optixmode" ; 
const char* Interactor::GUIMODE    = "gui" ; 



void Interactor::gui()
{
#ifdef GUI_
    ImGui::Text(" status: %s ", m_status );
#endif    
}





void Interactor::configureF(const char* name, std::vector<float> values)
{
    LOG(debug)<<"Interactor::configureF";
}

void Interactor::configureI(const char* name, std::vector<int> values)
{
    LOG(debug) << "Interactor::configureI";
    if(values.empty()) return ;
    if(strcmp(name, OPTIXMODE) == 0)
    {
        int last = values.back();
        setOptiXMode(last);
    }
}




void Interactor::setBookmarks(Bookmarks* bookmarks)
{
    m_bookmarks = bookmarks ;
}

void Interactor::setComposition(Composition* composition)
{
    m_composition = composition ;
    m_camera = composition->getCamera() ;
    m_view   = composition->getView();
    m_trackball = composition->getTrackball();
    m_clipper  = composition->getClipper();
}


void Interactor::cursor_drag(float x, float y, float dx, float dy, int ix, int iy )
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
        m_frame->touch(ix, iy );  
        // unProjects cursor position and identifiers smallest containing volume
        // frame records volume index of what is touched 
    }
}




void Interactor::key_pressed(unsigned int key)
{
    if(key < NUM_KEYS) m_keys_down[key] = true ; 

    if(key > 245) printf("Interactor::key_pressed %u \n", key );



    switch (key)
    {
        case GLFW_KEY_A:
            m_composition->nextMode(getModifiers()) ; 
            break;
        case GLFW_KEY_V:
            m_view->nextMode(getModifiers()) ; 
            break;
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
        case GLFW_KEY_G:
            m_gui_mode = !m_gui_mode ; 
            break;
        case GLFW_KEY_J:
            m_scene->jump(); 
            break;
        case GLFW_KEY_P:
            m_scene->nextPhotonStyle(); 
            break;
        case GLFW_KEY_Q:
            m_scene->toggleGeometry(); 
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
        case GLFW_KEY_C:
            m_clipper->next(); 
            break;
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
            number_key_pressed(key - GLFW_KEY_0);
            break; 
    } 
    updateStatus();
}




unsigned int Interactor::getModifiers()
{
    unsigned int modifiers = 0 ;
    if( m_keys_down[GLFW_KEY_LEFT_SHIFT]   || m_keys_down[GLFW_KEY_RIGHT_SHIFT] )    modifiers += e_shift ;
    if( m_keys_down[GLFW_KEY_LEFT_CONTROL] || m_keys_down[GLFW_KEY_RIGHT_CONTROL] )  modifiers += e_control ;
    if( m_keys_down[GLFW_KEY_LEFT_ALT]     || m_keys_down[GLFW_KEY_RIGHT_ALT] )      modifiers += e_option ;
    if( m_keys_down[GLFW_KEY_LEFT_SUPER]   || m_keys_down[GLFW_KEY_RIGHT_SUPER] )    modifiers += e_command ;
    return modifiers ; 
}


std::string Interactor::describeModifiers(unsigned int modifiers)
{
    std::stringstream ss ; 
    if(modifiers & e_shift)   ss << "shift " ; 
    if(modifiers & e_control) ss << "control " ; 
    if(modifiers & e_option)  ss << "option " ; 
    if(modifiers & e_command) ss << "command " ;
    return ss.str(); 
}

/*

 /usr/local/env/graphics/glfw/glfw-3.1.1/include/GLFW/glfw3.h 

                                                     mac keyboard
 382 #define GLFW_KEY_LEFT_SHIFT         340            "shift"
 383 #define GLFW_KEY_LEFT_CONTROL       341            "control"
 384 #define GLFW_KEY_LEFT_ALT           342            "option"
 385 #define GLFW_KEY_LEFT_SUPER         343            "command"
 386 #define GLFW_KEY_RIGHT_SHIFT        344
 387 #define GLFW_KEY_RIGHT_CONTROL      345
 388 #define GLFW_KEY_RIGHT_ALT          346
 389 #define GLFW_KEY_RIGHT_SUPER        347


*/


void Interactor::key_released(unsigned int key)
{
    if(key < NUM_KEYS) m_keys_down[key] = false ; 
    switch (key)
    {
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
            number_key_released(key - GLFW_KEY_0);
            break; 
    } 
}

void Interactor::number_key_pressed(unsigned int number)
{
    m_bookmark_mode = true ; 
    m_bookmarks->number_key_pressed(number, getModifiers());
}

void Interactor::number_key_released(unsigned int number)
{
    m_bookmarks->number_key_released(number);
    m_bookmark_mode = false ; 
}

void Interactor::updateStatus()
{
    snprintf(m_status, 64, "%s%s%s%s%s%s%s%s %10.3f %u ",
           m_zoom_mode ? "z" : "-",
           m_pan_mode  ? "x" : "-",
           m_far_mode  ? "f" : "-",
           m_near_mode ? "n" : "-",
           m_yfov_mode ? "y" : "-",
           m_rotate_mode ? "r" : "-",
           m_optix_mode ? "o" : "-",
           m_gui_mode ? "g" : "-",
           m_dragfactor,
           m_container 
           );
}

const char* Interactor::getStatus()
{
    return m_status ;
}

void Interactor::Print(const char* msg)
{
    updateStatus();
    printf("%s %s\n", msg, getStatus() );
}


