#include "GUI.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GLEQ.hh"

// npy-
#include "Index.hpp"
#include "NState.hpp"
#include "GLMFormat.hpp"

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
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"

// ggeo-
#include "GGeo.hh"
#include "GSurfaceLib.hh"
#include "GMaterialLib.hh"
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



GUI::GUI(GGeo* ggeo) 
   :
   m_ggeo(ggeo),
   m_show_test_window(false),
   m_bg_alpha(0.65f),
   m_scrub_alpha(0.01f),
   m_interactor(NULL),
   m_scene(NULL),
   m_composition(NULL),
   m_view(NULL),
   m_camera(NULL),
   m_clipper(NULL),
   m_trackball(NULL),
   m_bookmarks(NULL),
   m_state_gui(NULL),
   m_photons(NULL),
   m_animator(NULL)
{
}

void GUI::setInteractor(Interactor* interactor)
{
    m_interactor = interactor ; 
}
void GUI::setPhotons(Photons* photons)
{
    m_photons = photons ; 
}

void GUI::setView(View* view)
{
    m_view = view ; 
}

void GUI::setCamera(Camera* camera)
{
    m_camera = camera ; 
}
void GUI::setClipper(Clipper* clipper)
{
    m_clipper = clipper ; 
}
void GUI::setTrackball(Trackball* trackball)
{
    m_trackball = trackball ; 
}
void GUI::setBookmarks(Bookmarks* bookmarks)
{
    m_bookmarks = bookmarks ; 
}
void GUI::setStateGUI(StateGUI* state_gui)
{
    m_state_gui = state_gui ; 
}
void GUI::setAnimator(Animator* animator)
{
    m_animator = animator ; 
}


void GUI::setupHelpText(const std::string& txt)
{
    m_help = txt ; 
} 

void GUI::setupStats(const std::vector<std::string>& stats)
{
    m_stats = stats ; 
}
void GUI::setupParams(const std::vector<std::string>& params)
{
    m_params = params ; 
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


void GUI::composition_gui(Composition* composition)
{
    ImGui::SliderFloat( "lookPhi", composition->getLookAnglePtr(),  -180.f, 180.0f, "%0.3f");
    ImGui::SameLine();
    if(ImGui::Button("zeroPhi")) composition->setLookAngle(0.f) ;

    if(ImGui::Button("home")) composition->home();
    if(ImGui::Button("commit")) composition->commitView();


    std::string eye = composition->getEyeString();
    std::string look = composition->getLookString();
    std::string gaze = composition->getGazeString();
 
    ImGui::Text(" eye  : %s ", eye.c_str()); 
    ImGui::Text(" look : %s ", look.c_str()); 
    ImGui::Text(" gaze : %s ", gaze.c_str()); 

    glm::vec4 viewpoint = composition->getViewpoint();
    glm::vec4 lookpoint = composition->getLookpoint();
    glm::vec4 updir = composition->getUpdir();

    ImGui::Text(" viewpoint  : %s ", gformat(viewpoint).c_str()); 
    ImGui::Text(" lookpoint  : %s ", gformat(lookpoint).c_str()); 
    ImGui::Text(" updir      : %s ", gformat(updir).c_str()); 


    ImGui::Text(" setEyeGUI ");
    if(ImGui::Button(" +X")) composition->setEyeGUI(glm::vec3(1,0,0));
    ImGui::SameLine();
    if(ImGui::Button(" -X")) composition->setEyeGUI(glm::vec3(-1,0,0));
    ImGui::SameLine();
    if(ImGui::Button(" +Y")) composition->setEyeGUI(glm::vec3(0,1,0));
    ImGui::SameLine();
    if(ImGui::Button(" -Y")) composition->setEyeGUI(glm::vec3(0,-1,0));
    ImGui::SameLine();
    if(ImGui::Button(" +Z")) composition->setEyeGUI(glm::vec3(0,0,1));
    ImGui::SameLine();
    if(ImGui::Button(" -Z")) composition->setEyeGUI(glm::vec3(0,0,-1));


    float* param = composition->getParamPtr() ;
    ImGui::SliderFloat( "param.x", param + 0,  0.f, 1000.0f, "%0.3f", 2.0f);
    ImGui::SliderFloat( "param.y", param + 1,  0.f, 1.0f, "%0.3f", 2.0f );
    ImGui::SliderFloat( "z:alpha", param + 2,  0.f, 1.0f, "%0.3f");


    float* lpos = composition->getLightPositionPtr() ;
    ImGui::SliderFloat3( "lightposition", lpos,  -2.0f, 2.0f, "%0.3f");

    float* ldir = composition->getLightDirectionPtr() ;
    ImGui::SliderFloat3( "lightdirection", ldir,  -2.0f, 2.0f, "%0.3f");

    int* pickp = composition->getPickPtr() ;
    ImGui::SliderInt( "pick.x", pickp + 0,  1, 100 );  // modulo scale down
    ImGui::SliderInt( "pick.w", pickp + 3,  0, 1000000 );  // single photon pick

    int* colpar = composition->getColorParamPtr() ;
    ImGui::SliderInt( "colorparam.x", colpar + 0,  0, Composition::NUM_COLOR_STYLE  );  // record color mode
    ImGui::Text(" colorstyle : %s ", composition->getColorStyleName()); 

    int* np = composition->getNrmParamPtr() ;
    ImGui::SliderInt( "nrmparam.x", np + 0,  0, 1  );  
    ImGui::Text(" (nrm) normals : %s ",  *(np + 0) == 0 ? "NOT flipped" : "FLIPPED" );   

    ImGui::SliderInt( "nrmparam.z", np + 2,  0, 1  );  
    ImGui::Text(" (nrm) scanmode : %s ",  *(np + 2) == 0 ? "DISABLED" : "ENABLED" );   

    float* scanparam = composition->getScanParamPtr() ;
    ImGui::SliderFloat( "scanparam.x", scanparam + 0,  0.f, 1.0f, "%0.3f", 2.0f );
    ImGui::SliderFloat( "scanparam.y", scanparam + 1,  0.f, 1.0f, "%0.3f", 2.0f );
    ImGui::SliderFloat( "scanparam.z", scanparam + 2,  0.f, 1.0f, "%0.3f", 2.0f );
    ImGui::SliderFloat( "scanparam.w", scanparam + 3,  0.f, 1.0f, "%0.3f", 2.0f );

    *(scanparam + 0) = fmaxf( 0.0f , *(scanparam + 2) - *(scanparam + 3) ) ; 
    *(scanparam + 1) = fminf( 1.0f , *(scanparam + 2) + *(scanparam + 3) ) ; 

    ImGui::Text(" nrmparam.y geometrystyle : %s ", composition->getGeometryStyleName()); 

    glm::ivec4& pick = composition->getPick();
    ImGui::Text("pick %d %d %d %d ",
       pick.x, 
       pick.y, 
       pick.z, 
       pick.w);

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
        ImGui::Text("%s",m_help.c_str());
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Params"))
    {
        for(unsigned int i=0 ; i < m_params.size() ; i++) ImGui::Text("%s",m_params[i].c_str());
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Stats"))
    {
        for(unsigned int i=0 ; i < m_stats.size() ; i++) ImGui::Text("%s",m_stats[i].c_str());
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


    if (ImGui::CollapsingHeader("Composition"))
    {
       composition_gui(m_composition);
 
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



    OpticksAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames();
    if(qmat)
    {
        ImGui::Spacing();
        gui_item_index(qmat);
    } 

    OpticksAttrSeq* qsur = m_ggeo->getSurfaceLib()->getAttrNames();
    if(qsur)
    {
        ImGui::Spacing();
        gui_item_index(qsur);
    } 

    //OpticksAttrSeq* qflg = m_ggeo->getFlags()->getAttrIndex();
    OpticksAttrSeq* qflg = m_ggeo->getFlagNames();
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





void GUI::gui_item_index(OpticksAttrSeq* al)
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

           ImGui::TextColored(ImVec4(red/256.f,green/256.f,blue/256.f,1.0f), "%s", labels[i].c_str() );
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


