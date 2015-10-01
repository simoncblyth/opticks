#include "GUI.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

// npy-
#include "Index.hpp"

// ggeo-
#include "GLoader.hh"
#include "GItemIndex.hh"

// oglrap-
#include "Interactor.hh"
#include "Scene.hh"
#include "Composition.hh"
#include "Clipper.hh"
#include "View.hh"
#include "Camera.hh"
#include "Trackball.hh"
#include "Bookmarks.hh"
#include "Photons.hh"

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"


void GUI::setComposition(Composition* composition)
{
    m_composition = composition ; 
    setClipper(composition->getClipper());
    setView(composition->getView());
    setCamera(composition->getCamera());
    setTrackball(composition->getTrackball());
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
    m_composition->gui(); 

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("View"))
    {
        m_view->gui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Camera"))
    {
        m_camera->gui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Clipper"))
    {
        m_clipper->gui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Trackball"))
    {
        m_trackball->gui(); 
    }

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Bookmarks"))
    {
        m_bookmarks->gui(); 
    }


    ImGui::Spacing();
    if( m_photons )
    {
        m_photons->gui(); 
    }

    GItemIndex* materials = m_loader ? m_loader->getMaterials() : NULL ; 
    if(materials)
    {
        ImGui::Spacing();
        gui_item_index(materials);
    } 


    GItemIndex* surfaces = m_loader ? m_loader->getSurfaces() : NULL ; 
    if(surfaces)
    {
        ImGui::Spacing();
        gui_item_index(surfaces);
    } 

    GItemIndex* flags = m_loader ? m_loader->getFlags() : NULL ; 
    if(flags)
    {
        ImGui::Spacing();
        gui_item_index(flags);
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






void GUI::gui_item_index(GItemIndex* ii)
{
#ifdef GUI_
    Index* index = ii->getIndex(); 

    if (ImGui::CollapsingHeader(index->getItemType()))
    {   
       typedef std::vector<std::string> VS ; 
       typedef std::vector<unsigned int> VI ; 

       VS& labels = ii->getLabels();
       VI& codes = ii->getCodes();

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







