// /Users/blyth/env/graphics/glfw/glfwminimal/glfwminimal.cc
// http://www.glfw.org/docs/latest/quick.html

#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

#define SNAP 1 

#ifdef SNAP
#include "SPPM.hh"
struct Pix : SPPM 
{
    void download(); 
};

void Pix::download()
{
    glPixelStorei(GL_PACK_ALIGNMENT,1); // byte aligned output https://www.khronos.org/opengl/wiki/GLAPI/glPixelStore
    glReadPixels(0,0,pwidth,pheight,GL_RGBA, GL_UNSIGNED_BYTE, pixels );
    printf("snap\n");
}

Pix* pix = new Pix ; 
#endif




static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

#ifdef SNAP
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) pix->snap("/tmp/UseOpticksGLFWSPPM.ppm") ;
#endif

}

int main(void)
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(640, 480, "UseOpticksGLFWSPPM : SPACE to save snapshot, ESCAPE to exit", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
#ifdef SNAP
        pix->resize(width, height);
#endif

        ratio = width / (float) height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(-0.6f, -0.4f, 0.f);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.6f, -0.4f, 0.f);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0.f, 0.6f, 0.f);
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

