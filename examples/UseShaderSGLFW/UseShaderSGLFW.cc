/**
examples/UseShaderSGLFW/UseShaderSGLFW.cc
===========================================

::

    ~/o/examples/UseShaderSGLFW/go.sh 


* https://www.glfw.org/docs/latest/quick.html#quick_example

Started from ~/o/examples/UseShader and transitioned to 
using SGLFW.h to hide lots of boilerplate details.

**/


#include "SGLFW.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include <cstdlib>
#include <cstdio>
#include <iostream>

static const struct 
{
    float x, y, r, g, b ;
} 
vertices[3] =
{
    { -0.6f, -0.4f, 1.f, 0.f, 0.f },
    {  0.6f, -0.4f, 0.f, 1.f, 0.f },
    {   0.f,  0.6f, 0.f, 0.f, 1.f }
};


GLubyte indices[] = {
     0, 1, 2 
};
  

static const char* vertex_shader_text = R"glsl(
#version 410 core
uniform mat4 MVP;
in vec3 vCol;
in vec2 vPos;
out vec3 color;
void main()
{
    gl_Position = MVP * vec4(vPos, 0.0, 1.0);
    color = vCol;
}

)glsl";

static const char* geometry_shader_text = nullptr ; 
static const char* fragment_shader_text = R"glsl(
#version 410 core
in vec3 color;
out vec4 frag_color;

void main()
{
    frag_color = vec4(color, 1.0);
}

)glsl";

/**
gl_FragDepth = 0.1 ; 
gl_FragDepth = 0.5 ; 
    NB omitting a suitable setting for gl_FragDepth 
    when depth testing is enabled results in a black window.
    Depth testing is enabled in SGLFW::init with GL_DEPTH_TEST 
    and DepthFunc GL_LESS

**/



int main(void)
{
    SGLM gm ; 
    gm.setDepthTest(0);

    SGLFW gl(gm); 
    GLFWwindow* window = gl.window ; 

    SGLFW_VAO vao ; 
    vao.bind(); 

#ifdef WITH_INDICES
    SGLFW_Buffer ibuf( sizeof(indices), indices, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    ibuf.bind();
    ibuf.upload();
#endif

    assert( sizeof(vertices) == sizeof(float)*5*3 ); 
    SGLFW_Buffer vbuf( sizeof(vertices), vertices, GL_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    vbuf.bind();
    vbuf.upload();



    SGLFW_Program prog(nullptr, "vPos", nullptr, nullptr, nullptr, nullptr ); 
    prog.createFromText( vertex_shader_text, geometry_shader_text, fragment_shader_text); 

    GLint mvp_location = prog.getUniformLocation("MVP");   
    printf("mvp_location : %d \n", mvp_location ); 


//#define OLD_ATTRIB 1   

#ifdef OLD_ATTRIB

    printf("// OLD_ATTRIB \n" ); 
    GLint vpos_location = prog.getAttribLocation("vPos"); 
    GLint vcol_location = prog.getAttribLocation("vCol");
 
    printf("//vpos_location %d \n", vpos_location );  
    printf("//vcol_location %d \n", vcol_location );  
    
    glEnableVertexAttribArray(vpos_location); 
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*) 0);                   
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*) (sizeof(float) * 2)); 

#else
    printf("// NOT-OLD_ATTRIB \n" ); 
    prog.enableVertexAttribArray( "vPos", "2,GL_FLOAT,GL_FALSE,20,0,false" );  // 20 == sizeof(float)*5 stride in bytes
    prog.enableVertexAttribArray( "vCol", "3,GL_FLOAT,GL_FALSE,20,8,false" );  // 8 == sizeof(float)*2 offset in bytes 
    // getting two attrib from the same array via different size and offset 
#endif



    int count(0);
    int renderlooplimit(200); 
    bool exitloop(false); 

    while (!glfwWindowShouldClose(window) && !exitloop)
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glm::mat4 mvp = glm::scale(glm::mat4(1.0f), glm::vec3(1.5f));
        float* mvp_f = glm::value_ptr( mvp ); 

        prog.use(); 
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp_f );

         
#ifdef WITH_INDICES
        int indices_count = 3 ; 
        GLvoid* indices_offset = (GLvoid*)(sizeof(GLubyte) * 0) ; 
        glDrawElements(GL_TRIANGLES, indices_count, GL_UNSIGNED_BYTE, indices_offset );
#else
        //printf("//glDrawArrays\n");  
        glDrawArrays(GL_TRIANGLES, 0, 3 );
#endif

        glfwSwapBuffers(window);
        glfwPollEvents();
        exitloop = renderlooplimit > 0 && count++ > renderlooplimit ;   
    }
    exit(EXIT_SUCCESS);
}
