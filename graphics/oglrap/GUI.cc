#include "GUI.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

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
    setPhotons(scene->getPhotons());
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

void GUI::setupHelpText(std::string txt)
{
    m_help = txt ; 
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

    m_photons->gui(); 


    GItemIndex* materials = m_loader ? (GItemIndex*)m_loader->getMaterials() : NULL ; 
    if(materials)
    {
        ImGui::Spacing();
        materials->gui();
    } 


    GItemIndex* flags = m_loader ? (GItemIndex*)m_loader->getFlags() : NULL ; 
    if(flags)
    {
        ImGui::Spacing();
        flags->gui();
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

