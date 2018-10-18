// /Users/blyth/env/graphics/glfw/glfwminimal/glfwminimal.cc
// http://www.glfw.org/docs/latest/quick.html

#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

#define SNAP 1

#ifdef SNAP
struct Pix
{
    unsigned char* pixels ; 
    int pwidth ; 
    int pheight ; 

    Pix(int width, int height);

    void resize(int width, int height);
    void download();
    void save(const char* path);
    void snap(); 

    static void saveppm(const char* filename, int width, int height, unsigned char* image); 
};


Pix::Pix(int width, int height )
    :
    pixels(NULL),
    pwidth(width),
    pheight(height)
{
}

void Pix::resize( int width, int height )
{ 
    bool changed_size = !(width == pwidth && height == pheight) ; 
    if( pixels == NULL || changed_size )
    { 
        delete [] pixels ;
        pixels = NULL ; 
        pwidth = width ; 
        pheight = height ; 
        int size = 4 * pwidth * pheight ;
        pixels = new unsigned char[size];
        printf("creating resized pixels buffer (%d,%d) \n", pwidth, pheight ); 
    }
}
void Pix::download()
{
    glPixelStorei(GL_PACK_ALIGNMENT,1); // byte aligned output   https://www.khronos.org/opengl/wiki/GLAPI/glPixelStore
    glReadPixels(0,0,pwidth,pheight,GL_RGBA, GL_UNSIGNED_BYTE, pixels );
    printf("snap\n");
}
void Pix::save(const char* path)
{
    saveppm(path, pwidth, pheight, pixels );  
    printf("save\n");
}

void Pix::snap()
{
    download(); 
    save("/tmp/pix.ppm");  
}


void Pix::saveppm(const char* filename, int width, int height, unsigned char* image) {
    // save into ppm
    FILE * fp;
    fp = fopen(filename, "wb");

    int ncomp = 4;
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned char* data = new unsigned char[height*width*3] ;

    for( int y=height-1; y >= 0; --y ) // flip vertically
        {
            for( int x=0; x < width ; ++x )
                {
                    *(data + (y*width+x)*3+0) = image[(y*width+x)*ncomp+0] ;
                    *(data + (y*width+x)*3+1) = image[(y*width+x)*ncomp+1] ;
                    *(data + (y*width+x)*3+2) = image[(y*width+x)*ncomp+2] ;
                }
        }
    fwrite(data, sizeof(unsigned char)*height*width*3, 1, fp);
    fclose(fp);

    delete[] data;
}

Pix* pix = NULL ; 

#endif


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        printf("space\n"); 
        pix->snap(); 
    }

}



int main(void)
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        exit(EXIT_FAILURE);

    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

#ifdef SNAP
    pix = new Pix(0,0); 
#endif

    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        pix->resize(width, height);

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

