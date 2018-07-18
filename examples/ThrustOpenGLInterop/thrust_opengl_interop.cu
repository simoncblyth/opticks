//
//  Started from glfwminimal-
//  and attempting to bring in Thrust OpenGL interop 
//  following https://gist.github.com/dangets/2926425/download#
//

#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>

#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


unsigned int g_window_width = 512;
unsigned int g_window_height = 512;

unsigned int g_mesh_width = 256;
unsigned int g_mesh_height = 256;


thrust::device_ptr<float4> dev_ptr;
GLuint vbo;
struct cudaGraphicsResource *vbo_cuda;

float g_anim = 0.0;

// mouse controls
int g_mouse_old_x, g_mouse_old_y;
int g_mouse_buttons = 0;
float g_rotate_x = 0.0, g_rotate_y = 0.0;
float g_translate_z = -3.0;


struct sine_wave
{
    sine_wave(unsigned int w, unsigned int h, float t)
        : 
        width(w), 
        height(h), 
        time(t) 
    {
    }

    __host__ __device__
    float4 operator()(unsigned int i)
    {
        unsigned int x = i % width;
        unsigned int y = i / width;

        // calculate uv coordinates
        float u = x / (float) width;
        float v = y / (float) height;
        u = u*2.0f - 1.0f;
        v = v*2.0f - 1.0f;

        // calculate simple sine wave pattern
        float freq = 4.0f;
        float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

        // write output vertex
        return make_float4(u, w, v, 1.0f);
    }

    float time;
    unsigned int width, height;
};


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void init()
{
    //cudaGLSetGLDevice(0);

    unsigned int size = g_mesh_width * g_mesh_height * sizeof(float4);
    printf("init size:%d width:%d height:%d \n", size, g_mesh_width, g_mesh_height); 

    float* data = new float[size] ; 
    unsigned int index(0);
    for(unsigned ix=0 ; ix < g_mesh_width  ; ix++){
    for(unsigned iy=0 ; iy < g_mesh_height ; iy++){
        index = iy*g_mesh_width + ix ; 
        data[index*4+0] = float(ix)/float(g_mesh_width) ;   
        data[index*4+1] = float(iy)/float(g_mesh_height) ;   
        data[index*4+2] = 0.f ;   
        data[index*4+3] = 1.f ;   
    }
    }


    // create vbo
    glGenBuffers(1, &vbo);
    // bind, initialize, unbind
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);  // target, size, data, usage
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    //cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);
}


void setup_view(float ratio)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void display_triangles()
{
    glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
    glBegin(GL_TRIANGLES);
    glColor3f(1.f, 0.f, 0.f);
    glVertex3f(-0.6f, -0.4f, 0.f);
    glColor3f(0.f, 1.f, 0.f);
    glVertex3f(0.6f, -0.4f, 0.f);
    glColor3f(0.f, 0.f, 1.f);
    glVertex3f(0.f, 0.6f, 0.f);
    glEnd();
}


void display_thrust(bool thrust)
{
    if(thrust)
    {
        cudaGraphicsMapResources(1, &vbo_cuda, 0);

        float4 *raw_ptr;
        size_t buf_size;
        cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);
        dev_ptr = thrust::device_pointer_cast(raw_ptr);

        // transform the mesh
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last(g_mesh_width * g_mesh_height);
        thrust::transform(first, last, dev_ptr, sine_wave(g_mesh_width, g_mesh_height, g_anim));
        cudaGraphicsUnmapResources(1, &vbo_cuda, 0);
    }


    glTranslatef(0.0, 0.0, g_translate_z);
    glRotatef(g_rotate_x, 1.0, 0.0, 0.0);
    glRotatef(g_rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0); // size, type, stride, pointer

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_LINES, 0, g_mesh_width * g_mesh_height); // mode, first, count
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //glutSwapBuffers();
    //glutPostRedisplay();

    g_anim += 0.001;
}






int main(int argc, char** argv)
{

    char mode = argc > 1 && strlen(argv[1]) > 0 ? argv[1][0] : 'A' ; 

    const char* msg = NULL ; 
    switch(mode)
    {
       case 'A':msg="mode A : display_triangles     " ; break ;
       case 'B':msg="mode B : display_thrust(true)  " ; break ;
       case 'C':msg="mode C : display_thrust(false) " ; break ;
        default:msg="unknown mode " ; break ;  
    }

    printf(" %s : %s \n", argv[0], msg );


    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

    window = glfwCreateWindow(g_window_width, g_window_height, msg, NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    init();


    glfwSetKeyCallback(window, key_callback);
    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
                 
        glViewport(0, 0, width, height);

        setup_view(ratio);

        switch(mode)
        {
           case 'A':display_triangles() ;break;
           case 'B':display_thrust(true) ;break;
           case 'C':display_thrust(false) ;break;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}




