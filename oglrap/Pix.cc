#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "PLOG.hh" 
#include "Pix.hh" 


void Pix::download()
{
    glPixelStorei(GL_PACK_ALIGNMENT,1); // byte aligned output https://www.khronos.org/opengl/wiki/GLAPI/glPixelStore
    glReadPixels(0,0,pwidth*pscale,pheight*pscale,GL_RGBA, GL_UNSIGNED_BYTE, pixels );

    LOG(info) 
        << desc()
        ;


}


