#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"


const char* vertSrc = R"glsl(

    #version 410 core

    layout(location = 0) in vec4 vPosition ;
    void main()
    {
        gl_Position = vPosition ;  
    }

)glsl";


const char* fragSrc = R"glsl(

    #version 410 core
   
    out vec4 fColor ; 

    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }

)glsl";


struct V { float x,y,z,w ; };

static const unsigned NUM_VERT = 3 ; 

V verts[NUM_VERT] = {
    { -1.f , -1.f,  0.f,  1.f }, 
    { -1.f ,  1.f,  0.f,  1.f },
    {  1.f ,  0.f,  0.f,  1.f }
};


int main()
{
    Frame frame ; 

    Prog prog(vertSrc, NULL, fragSrc ) ; 
    prog.compile();
    prog.create();
    prog.link();

    GLint vPosition = prog.getAttribLocation("vPosition");
    std::cout << "vPosition " << vPosition << std::endl ; 


    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    std::cout << "vao " << vao << std::endl ; 

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(vPosition);   
    glVertexAttribPointer(vPosition, NUM_VERT, GL_FLOAT, GL_FALSE, 0, 0);
    // https://alfonse.bitbucket.io/oldtut/Basics/Tut01%20Dissecting%20Display.html

    while (!glfwWindowShouldClose(frame.window))
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLES, 0, NUM_VERT );
        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    prog.destroy();

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    frame.destroy();

    exit(EXIT_SUCCESS);
}


