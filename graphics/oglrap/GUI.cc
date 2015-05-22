#include "GUI.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"


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



/*
void GUI::choose( std::vector<std::pair<int, std::string> >& choices, std::vector<int>& selection )
{
    for(unsigned int i=0 ; i < choices.size() ; i++)
    {
        std::pair<int, std::string> choice = choices[i];
        if(ImGui::Button(choice.second.c_str())) selection[i] ^= 1 ; 
    }
}
*/


void GUI::choose( std::vector<std::pair<int, std::string> >& choices, bool* selection )
{
    for(unsigned int i=0 ; i < choices.size() ; i++)
    {
        std::pair<int, std::string> choice = choices[i];
        ImGui::Checkbox(choice.second.c_str(), selection+i );
    }
}

// follow pattern of ImGui::ShowTestWindow
void GUI::show(bool* opened)
{
    static float bg_alpha = 0.65f;
    ImGuiWindowFlags window_flags = 0;

    if (!ImGui::Begin("GGeoView", opened, ImVec2(550,680), bg_alpha, window_flags)) 
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return ; 
    }

    ImGui::PushItemWidth(-140);  
    ImGui::Text("ImGui says hello.");

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Photon Boundary Selection"))
    {

    }



}



void GUI::demo()
{
    // 1. Show a simple window
    // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
    {
            ImGui::Text("Hello, world!");
            ImGui::SliderFloat("float", &m_f, 0.0f, 1.0f);
            if (ImGui::Button("Test Window")) m_show_test_window ^= 1;
            if (ImGui::Button("Another Window")) m_show_another_window ^= 1;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    }

    if (m_show_another_window)
    {
        ImGui::SetNextWindowSize(ImVec2(200,100), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Another Window", &m_show_another_window);
        ImGui::Text("Hello");
        ImGui::End();
    }

    // 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
    if (m_show_test_window)
    {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
        ImGui::ShowTestWindow(&m_show_test_window);
    }
}

void GUI::render()
{
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    ImGui::Render();

    // https://github.com/ocornut/imgui/issues/109
    // fix ImGui diddling of OpenGL state
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

void GUI::shutdown()
{
    ImGui_ImplGlfwGL3_Shutdown();
}

GUI::~GUI()
{
}

