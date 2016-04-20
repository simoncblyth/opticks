#include "GUI.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

// npy-
#include "Index.hpp"
#include "NState.hpp"

// opticks-
#include "Opticks.hh"
#include "Clipper.hh"
#include "Camera.hh"
#include "Animator.hh"
#include "Trackball.hh"
#include "View.hh"
#include "TrackView.hh"
#include "OrbitalView.hh"
#include "InterpolatedView.hh"

// ggeo-
#include "GGeo.hh"
#include "GSurfaceLib.hh"
#include "GMaterialLib.hh"
#include "GFlags.hh"
#include "GAttrSeq.hh"
#include "GItemIndex.hh"

// oglrap-
#include "Interactor.hh"
#include "Scene.hh"

#include "Composition.hh"
#include "Bookmarks.hh"

#include "Photons.hh"
#include "StateGUI.hh"



#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"


void GUI::setComposition(Composition* composition)
{
    m_composition = composition ; 
    setClipper(composition->getClipper());
    setView(composition->getView());
    setCamera(composition->getCamera());
    setTrackball(composition->getTrackball());
    setAnimator(NULL); // defer
}



void GUI::setScene(Scene* scene)
{
    m_scene = scene ;
}


void GUI::init(GLFWwindow* window)
{
    bool install_callbacks = false ; 
    ImGui_ImplGlfwGL3_Init(window, install_callbacks );

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "/tmp/imgui.ini";
}

void GUI::newframe()
{
    ImGui_ImplGlfwGL3_NewFrame();
}

void GUI::choose( std::vector<std::pair<int, std::string> >* choices, bool* selection )
{
    for(unsigned int i=0 ; i < choices->size() ; i++)
    {
        std::pair<int, std::string> choice = (*choices)[i];
        ImGui::Checkbox(choice.second.c_str(), selection+i );
    }
}

void GUI::choose( unsigned int n, const char** choices, bool** selection )
{
    for(unsigned int i=0 ; i < n ; i++)
    {
        ImGui::Checkbox(choices[i], selection[i]);
    }
}



void GUI::show_scrubber(bool* opened)
{
    if(!m_animator) m_animator = m_composition->getAnimator();

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar ;

    if (!ImGui::Begin("Scrubber", opened, ImVec2(550,100), m_scrub_alpha, window_flags)) 
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return ; 
    }

    ImGui::PushItemWidth(-140);  

    if(m_animator)
    {
        animator_gui(m_animator, "time (ns)", "%0.3f", 2.0f);
    } 

    //ImGui::SliderFloat("float", &m_scrub_alpha, 0.0f, 1.0f);

    ImGui::End();
}



bool GUI::animator_gui(Animator* animator, const char* label, const char* fmt, float power)
{
    Animator::Mode_t prior = animator->getMode() ;

    float* target = animator->getTarget();
    float low = animator->getLow();
    float high = animator->getHigh();
    float fraction = animator->getFractionForValue(*target);
    int* mode = animator->getModePtr() ;   // address of enum cast to int*

    ImGui::SliderFloat( label, target, low, high , fmt, power);
    ImGui::Text("animation mode: ");

    ImGui::RadioButton( Animator::OFF_ , mode, Animator::OFF); ImGui::SameLine();

    if(animator->isSlowEnabled())
    {
        ImGui::RadioButton(Animator::SLOW_, mode, Animator::SLOW);
        ImGui::SameLine(); 
    }
    if(animator->isNormEnabled())
    {
        ImGui::RadioButton(Animator::NORM_, mode, Animator::NORM);
        ImGui::SameLine();
    }
    if(animator->isFastEnabled())
    {
        ImGui::RadioButton(Animator::FAST_, mode, Animator::FAST); //ImGui::SameLine();
    }

    
    if(animator->isModeChanged(prior))
    {
       animator->modeTransition(fraction);
    }

    return animator->isActive() ;
}


void GUI::standard_view(View* view)
{
    if(ImGui::Button("home")) view->home();
    ImGui::SliderFloat3("eye",  view->getEyePtr(),  -1.0f, 1.0f);
    ImGui::SliderFloat3("look", view->getLookPtr(), -1.0f, 1.0f);
    ImGui::SliderFloat3("up",   view->getUpPtr(), -1.0f, 1.0f);
}

void GUI::track_view(TrackView* tv)
{
    Animator* animator = tv->getAnimator();
    if(animator)
    {   
         animator_gui(animator, "TrackView ", "%0.3f", 2.0f);
    }   
    ImGui::SliderFloat("tmin offset (ns)", tv->getTMinOffsetPtr(), -20.0f, 20.0f);
    ImGui::SliderFloat("tmax offset (ns)", tv->getTMaxOffsetPtr(), -20.0f, 20.0f);
    ImGui::SliderFloat("teye offset (ns)", tv->getTEyeOffsetPtr(), -20.0f, 50.0f);
    ImGui::SliderFloat("tlook offset (ns)",tv->getTLookOffsetPtr(), -20.0f, 50.0f);
    ImGui::SliderFloat("fraction scale",   tv->getFractionScalePtr(), 1.0f, 2.0f);
}

void GUI::orbital_view(OrbitalView* ov)
{
    Animator* animator = ov->getAnimator();
    if(animator)
    {   
         animator_gui(animator, "OrbitalView ", "%0.3f", 2.0f);
         ImGui::Text(" fraction %10.3f ", animator->getFractionFromTarget()  );  
    }   
}

void GUI::interpolated_view(InterpolatedView* iv)
{
    Animator* animator = iv->getAnimator();
    if(animator)
    {
         animator_gui(animator, "InterpolatedView ", "%0.3f", 2.0f);
         ImGui::Text(" fraction %10.3f ", animator->getFractionFromTarget()  );
    }
}




void GUI::viewgui()
{
    if(m_view->isTrack())
    {
         TrackView* tv = dynamic_cast<TrackView*>(m_view) ;
         track_view(tv);
    } 
    else if(m_view->isOrbital())
    {
         OrbitalView* ov = dynamic_cast<OrbitalView*>(m_view) ;
         orbital_view(ov);
    }
    else if(m_view->isInterpolated())
    {
         InterpolatedView* iv = dynamic_cast<InterpolatedView*>(m_view) ;
         interpolated_view(iv); 
    }
    else if(m_view->isStandard())
    {
         standard_view(m_view); 
    }
}


void GUI::camera_gui(Camera* camera)
{
    float power = 2.0f ; 
    ImGui::SliderFloat("near", camera->getNearPtr(), camera->getNearMin(), camera->getNearMax(), "%.3f", power );  
    ImGui::SliderFloat("far",  camera->getFarPtr(),  camera->getFarMin(),  camera->getFarMax() , "%.3f", power );
    ImGui::SliderFloat("zoom", camera->getZoomPtr(), camera->getZoomMin(), camera->getZoomMax(), "%.3f", power);
    ImGui::SliderFloat("scale",camera->getScalePtr(),camera->getScaleMin(),camera->getScaleMax(), "%.3f", power);
    ImGui::Checkbox("parallel", camera->getParallelPtr() );
    if (ImGui::Button("Camera Summary")) camera->Summary();
}


void GUI::trackball_gui(Trackball* trackball)
{
    if (ImGui::Button("Home")) trackball->home();
    if (ImGui::Button("Summary")) trackball->Summary();
    ImGui::SliderFloat3("translate",  trackball->getTranslationPtr(), trackball->getTranslationMin(), trackball->getTranslationMax() );
    ImGui::SliderFloat("radius",   trackball->getRadiusPtr(), trackball->getRadiusMin(), trackball->getRadiusMax() );
    ImGui::SliderFloat("tfactor",  trackball->getTFactorPtr(),  trackball->getTFactorMin(), trackball->getTFactorMax() );
    ImGui::Text(" quat: %s", trackball->getOrientationString().c_str() );
}



void GUI::clipper_gui(Clipper* clipper)
{
    // TODO: cut 2 degrees of freedom 
    // point and direction overspecifies plane, causing whacky interface
    // just need a scalar along the normal 

    ImGui::SliderFloat3("point",  clipper->getPointPtr(),  -1.0f, 1.0f);
    ImGui::SliderFloat3("normal", clipper->getNormalPtr(), -1.0f, 1.0f);
    //ImGui::SliderFloat3("absplane", getPlanePtr(), -1.0f, 1.0f);
}



void GUI::bookmarks_gui(Bookmarks* bookmarks)
{ 
    ImGui::SameLine();
    if(ImGui::Button("collect")) bookmarks->collect();
    ImGui::SameLine();
    if(ImGui::Button("apply")) bookmarks->apply();

    ImGui::SliderInt( "IVperiod", bookmarks->getIVPeriodPtr(),  50, 400 ); 

    int* curr = bookmarks->getCurrentPtr();
    int* curr_gui = bookmarks->getCurrentGuiPtr();

    for(Bookmarks::MUSI it=bookmarks->begin() ; it!=bookmarks->end() ; it++)
    {
         unsigned int num = it->first ; 
         std::string name = NState::FormName(num) ; 
         ImGui::RadioButton(name.c_str(), curr_gui, num);
    }

    // not directly setting m_current as need to notice a change
    if(*curr_gui != *curr ) 
    {
        bookmarks->setCurrent(*curr_gui);
        ImGui::Text(" changed : %d ", bookmarks->getCurrent());
        bookmarks->apply();
    }
}



// follow pattern of ImGui::ShowTestWindow
void GUI::show(bool* opened)
{
    if (m_show_test_window)
    {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
        ImGui::ShowTestWindow(&m_show_test_window);
    }

    ImGuiWindowFlags window_flags = 0;

    if (!ImGui::Begin("GGeoView", opened, ImVec2(550,680), m_bg_alpha, window_flags)) 
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return ; 
    }

    ImGui::PushItemWidth(-140);  

    ImGui::SliderFloat("float", &m_bg_alpha, 0.0f, 1.0f);

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Help"))
    {
        ImGui::Text(m_help.c_str());
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Params"))
    {
        for(unsigned int i=0 ; i < m_params.size() ; i++) ImGui::Text(m_params[i].c_str());
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Stats"))
    {
        for(unsigned int i=0 ; i < m_stats.size() ; i++) ImGui::Text(m_stats[i].c_str());
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Interactor"))
    {
        m_interactor->gui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Scene"))
    {
        m_scene->gui(); 
    }

    ImGui::Spacing();

    {
       m_composition->gui(); 
       Animator* animator = m_composition->getAnimator(); 

       if(animator)
       {
           animator_gui(animator, "time (ns)", "%0.3f", 2.0f);
           float* target = animator->getTarget();
           ImGui::Text(" time (ns) * %10.3f (mm/ns) : %10.3f mm ", Opticks::F_SPEED_OF_LIGHT, *target * Opticks::F_SPEED_OF_LIGHT );
       }  
    }



    ImGui::Spacing();
    if (ImGui::CollapsingHeader("View"))
    {
        viewgui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Camera"))
    {
        camera_gui(m_camera); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Clipper"))
    {
        clipper_gui(m_clipper); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Trackball"))
    {
        trackball_gui(m_trackball); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Bookmarks"))
    {
        bookmarks_gui(m_bookmarks); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("State"))
    {
        m_state_gui->gui(); 
    }




    ImGui::Spacing();
    if( m_photons )
    {
        m_photons->gui(); 
    }



    GAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames();
    if(qmat)
    {
        ImGui::Spacing();
        gui_item_index(qmat);
    } 

    GAttrSeq* qsur = m_ggeo->getSurfaceLib()->getAttrNames();
    if(qsur)
    {
        ImGui::Spacing();
        gui_item_index(qsur);
    } 

    GAttrSeq* qflg = m_ggeo->getFlags()->getAttrIndex();
    if(qflg)
    {
        ImGui::Spacing();
        gui_item_index(qflg);
    } 





    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Dev"))
    {
        ImGui::Checkbox("ImGui::ShowTestWindow", &m_show_test_window);
    }

    ImGui::End();
}






void GUI::render()
{
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    ImGui::Render();

    // https://github.com/ocornut/imgui/issues/109
    // fix ImGui diddling of OpenGL state
    glDisable(GL_BLEND);
    //glEnable(GL_CULL_FACE);  going-one-sided causes issues
    glEnable(GL_DEPTH_TEST);
}

void GUI::shutdown()
{
    ImGui_ImplGlfwGL3_Shutdown();
}

GUI::~GUI()
{
}





void GUI::gui_item_index(GAttrSeq* al)
{
    gui_item_index( al->getType(), al->getLabels(), al->getColorCodes());
}

//void GUI::gui_item_index(GItemIndex* ii)
//{
//    Index* idx = ii->getIndex(); 
//    gui_item_index( idx->getItemType(), ii->getLabels(), ii->getCodes());
//}


void GUI::gui_item_index(const char* type, std::vector<std::string>& labels, std::vector<unsigned int>& codes)
{
#ifdef GUI_
    if (ImGui::CollapsingHeader(type))
    {   
       assert(labels.size() == codes.size());
       for(unsigned int i=0 ; i < labels.size() ; i++)
       {   
           unsigned int code = codes[i] ;
           unsigned int red   = (code & 0xFF0000) >> 16 ;
           unsigned int green = (code & 0x00FF00) >>  8 ; 
           unsigned int blue  = (code & 0x0000FF)  ;

           ImGui::TextColored(ImVec4(red/256.,green/256.,blue/256.,1.0f), labels[i].c_str() );
       }   
    }   
#endif
}



void GUI::gui_radio_select(GItemIndex* ii)
{
#ifdef GUI_
    typedef std::vector<std::string> VS ; 
    Index* index = ii->getIndex(); 

    if (ImGui::CollapsingHeader(index->getTitle()))
    {   
       VS& labels = ii->getLabels();
       VS  names = index->getNames();
       assert(names.size() == labels.size());

       int* ptr = index->getSelectedPtr();

       std::string all("All ");
       all += index->getItemType() ;   

       ImGui::RadioButton( all.c_str(), ptr, 0 );

       for(unsigned int i=0 ; i < labels.size() ; i++)
       {   
           std::string iname = names[i] ;
           std::string label = labels[i] ;
           unsigned int local  = index->getIndexLocal(iname.c_str()) ;
           ImGui::RadioButton( label.c_str(), ptr, local);
       }   
       ImGui::Text("%s %d ", index->getItemType(), *ptr);
   }   
#endif
}


