
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "gleq.h"

#include "app.hh"
#include "Scene.hh"
#include "Config.hh"

//#include <boost/bind.hpp>


void _update_fps_counter (GLFWwindow* window) {
  static double previous_seconds = glfwGetTime ();
  static int frame_count;
  double current_seconds = glfwGetTime ();
  double elapsed_seconds = current_seconds - previous_seconds;
  if (elapsed_seconds > 0.25) {
    previous_seconds = current_seconds;
    double fps = (double)frame_count / elapsed_seconds;
    char tmp[128];
    sprintf (tmp, "opengl @ fps: %.2f", fps);
    glfwSetWindowTitle (window, tmp);
    frame_count = 0;
  }
  frame_count++;
}


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}


App::App(Config* config) : 
     m_config(config),
     m_title(NULL),
     m_window(NULL),
     m_scene(NULL)
//     m_eventQueue(),
//     m_udpServer(m_ioService,m_config->getUdpPort(),&m_eventQueue,&m_eventQueueMutex)
{
     m_config->dump();
}

App::~App()
{
    free((void*)m_title);
}


void App::setSize(unsigned int width, unsigned int height)
{
    m_width = width ;
    m_height = height ;
}
void App::setTitle(const char* title)
{
    m_title = strdup(title);
}
void App::setScene(Scene* scene)
{
    m_scene = scene ;
}


void App::init()
{
   // init_net();
    init_graphics();
}

//void App::init_net()
//{
//    m_netThread = new boost::thread(boost::bind(&boost::asio::io_service::run, &m_ioService));
//}


void App::init_graphics()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) ::exit(EXIT_FAILURE);

#ifdef  __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    m_window = glfwCreateWindow(m_width, m_height, m_title, NULL, NULL);
    if (!m_window)
    {
        glfwTerminate();
        ::exit(EXIT_FAILURE);
    }

    // hookup the callbacks and arranges outcomes into event queue 
    gleqTrackWindow(m_window);

    glfwMakeContextCurrent(m_window);

    // start GLEW extension handler, segfaults if done before glfwCreateWindow
    glewExperimental = GL_TRUE;
    glewInit ();

    glfwSwapInterval(1);  // vsync hinting

    // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    printf ("Renderer: %s\n", renderer);
    printf ("OpenGL version supported %s\n", version);

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    printf("FramebufferSize %d %d \n", width, height);    
}


void App::listen()
{
    glfwPollEvents();

    GLEQevent event;
    while (gleqNextEvent(&event))
    {
        dump_event(event);
        handle_event(event);
        gleqFreeEvent(&event);
    }
}
 
void App::on_msg(std::string msg)
{
    printf("App::on_msg [%s]\n", msg.c_str());
}

void App::on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
{
    printf("App::on_npy [%s] shape %d,%d,%d \n", metadata.c_str(), shape[0], shape[1], shape[2]);
}




/*
void App::listenUDP()
{
    boost::mutex::scoped_lock lock(m_eventQueueMutex);

    std::vector<std::string> lines ;
    while (!m_eventQueue.isEmpty()) 
    {
        EventQueueItem_t tmp;
        tmp = m_eventQueue.pop();
        std::string msg(tmp.begin(), tmp.end());
        lines.push_back(msg);
        printf("App::listenUDP [%s]\n", msg.c_str());
    }

    m_config->parse(lines, ' ');
}
*/




void App::runloop()
{
    while (!glfwWindowShouldClose(m_window))
    {
        listen(); 
        //listenUDP(); 
        //poll_one();  // give net_app a few cycles, to complete posts from the net thread
        render();
        glfwSwapBuffers(m_window);
    }
}


void App::render()
{
     _update_fps_counter (m_window);

     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     glViewport(0, 0, m_width, m_height);

     m_scene->draw(m_width, m_height);
}


void App::resize(unsigned int width, unsigned int height)
{
     setSize(width, height);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     glViewport(0, 0, m_width, m_height);
}


void App::handle_event(GLEQevent& event)
{
    switch (event.type)
    {
        case GLEQ_FRAMEBUFFER_RESIZED:
            printf("Framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height);
            break;
        case GLEQ_WINDOW_MOVED:
        case GLEQ_WINDOW_RESIZED:
            printf("Window resized to (%i %i)\n", event.size.width, event.size.height);
            resize(event.size.width, event.size.height);
            break;
        case GLEQ_WINDOW_CLOSED:
        case GLEQ_WINDOW_REFRESH:
        case GLEQ_WINDOW_FOCUSED:
        case GLEQ_WINDOW_DEFOCUSED:
        case GLEQ_WINDOW_ICONIFIED:
        case GLEQ_WINDOW_RESTORED:
        case GLEQ_BUTTON_PRESSED:
        case GLEQ_BUTTON_RELEASED:
        case GLEQ_CURSOR_MOVED:
        case GLEQ_CURSOR_ENTERED:
        case GLEQ_CURSOR_LEFT:
        case GLEQ_SCROLLED:
        case GLEQ_KEY_PRESSED:
             key_pressed(event.key.key );
             break;

        case GLEQ_KEY_REPEATED:
        case GLEQ_KEY_RELEASED:
        case GLEQ_CHARACTER_INPUT:
        case GLEQ_FILE_DROPPED:
        case GLEQ_NONE:
            break;
    }
} 

void App::key_pressed(unsigned int key)
{
    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            printf("escape\n");
            glfwSetWindowShouldClose (m_window, 1);
            break;
    }
}  
 
void App::dump_event(GLEQevent& event)
{
    switch (event.type)
    {
        case GLEQ_WINDOW_MOVED:
            printf("Window moved to (%.0f %.0f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_WINDOW_RESIZED:
            printf("Window resized to (%i %i)\n", event.size.width, event.size.height);
            break;
        case GLEQ_WINDOW_CLOSED:
            printf("Window close request\n");
            break;
        case GLEQ_WINDOW_REFRESH:
            printf("Window refresh request\n");
            break;
        case GLEQ_WINDOW_FOCUSED:
            printf("Window focused\n");
            break;
        case GLEQ_WINDOW_DEFOCUSED:
            printf("Window defocused\n");
            break;
        case GLEQ_WINDOW_ICONIFIED:
            printf("Window iconified\n");
            break;
        case GLEQ_WINDOW_RESTORED:
            printf("Window restored\n");
            break;
        case GLEQ_FRAMEBUFFER_RESIZED:
            printf("Framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
            break;
        case GLEQ_BUTTON_PRESSED:
            printf("Button %i pressed\n", event.button.button);
            break;
        case GLEQ_BUTTON_RELEASED:
            printf("Button %i released\n", event.button.button);
            break;
        case GLEQ_CURSOR_MOVED:
            printf("Cursor moved to (%0.2f %0.2f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_CURSOR_ENTERED:
            printf("Cursor entered window\n");
            break;
        case GLEQ_CURSOR_LEFT:
            printf("Cursor left window\n");
            break;
        case GLEQ_SCROLLED:
            printf("Scrolled (%0.2f %0.2f)\n", event.pos.x, event.pos.y);
            break;
        case GLEQ_KEY_PRESSED:
            printf("Key 0x%02x pressed\n", event.key.key);
            break;
        case GLEQ_KEY_REPEATED:
            printf("Key 0x%02x repeated\n", event.key.key);
            break;
        case GLEQ_KEY_RELEASED:
            printf("Key 0x%02x released\n", event.key.key);
            break;
        case GLEQ_CHARACTER_INPUT:
            printf("Character 0x%08x input\n", event.character.codepoint);
            break;
        case GLEQ_FILE_DROPPED:
            printf("%i files dropped\n", event.file.count);
            for (int i = 0;  i < event.file.count;  i++)
                printf("\t%s\n", event.file.paths[i]);
            break;
        case GLEQ_NONE:
            break;
    }
}



void App::exit()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}


