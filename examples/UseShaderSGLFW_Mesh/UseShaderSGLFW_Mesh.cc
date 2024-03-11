/**
examples/UseShaderSGLFW/UseShaderSGLFW_Mesh.cc
================================================

Started from ~/o/examples/UseShaderSGLFW and 
transitioned from single triangle to a mesh.::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=wireframe ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    SHADER=normal ~/o/examples/UseShaderSGLFW_Mesh/go.sh 

See also::

    ~/o/u4/tests/U4Mesh_test.sh 

    SOLID=Cons ~/o/u4/tests/U4Mesh_test.sh ana
    SOLID=Tubs ~/o/u4/tests/U4Mesh_test.sh ana

    ~/o/sysrap/tests/SMesh_test.sh 
    SOLID=Tet ~/o/sysrap/tests/SMesh_test.sh run


* https://www.khronos.org/opengl/wiki/Face_Culling

* https://learnopengl.com/Advanced-OpenGL/Face-culling

When we define a set of triangle vertices we're defining them in a certain
winding order that is either clockwise or counter-clockwise. Each triangle
consists of 3 vertices and we specify those 3 vertices in a winding order as
seen from the center of the triangle. 

::

             
            + 0           


      2 +   + 1
     

      0->1->2 CW
      0->2->1 CCW

Each set of 3 vertices that form a triangle primitive thus contain a winding
order. OpenGL uses this information when rendering your primitives to determine
if a triangle is a front-facing or a back-facing triangle. By default,
triangles defined with counter-clockwise vertices are processed as front-facing
triangles.

When defining your vertex order you visualize the corresponding triangle as if
it was facing you, so each triangle that you're specifying should be
counter-clockwise as if you're directly facing that triangle. The cool thing
about specifying all your vertices like this is that the actual winding order
is calculated at the rasterization stage, so when the vertex shader has already
run. The vertices are then seen as from the viewer's point of view. 









**/

#include "SGLFW.h"
#include "SGLM.h"
#include "SMesh.h"
#include "ssys.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 
    sframe fr ; fr.ce = make_float4(mesh->ce.x, mesh->ce.y, mesh->ce.z, mesh->ce.w ); 
    SGLM gm ; 
    //gm.setLookRotation( 45.f , {1.f, 1.f, 1.f } );  // angleAxis quaternion 
    gm.set_frame(fr) ; std::cout << gm.desc() ;  // HMM: set_ce ? avoid frame when not needed ?

    SGLFW gl(gm, gm.Width(), gm.Height(), mesh->name ); 
    gl.createProgram("$SHADER_FOLD"); 

    float* MVP_ptr = gm.MVP_ptr ;
    //float* MVP_ptr = gm.IDENTITY_ptr ;
    //float* MVP_ptr = gm.MV_ptr ;

    gl.locateMVP("MVP",  MVP_ptr ); 

    //glEnable(GL_CULL_FACE); 
    //glCullFace(GL_FRONT);  

    SGLFW_VAO vao ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    SGLFW_Buffer ibuf( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 

    SGLFW_Buffer vbuf( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    gl.enableAttrib( "vPos", "3,GL_FLOAT,GL_FALSE,12,0,false" );  // 3:vec3, 12:byte_stride 0:byte_offet

    SGLFW_Buffer nbuf( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    gl.enableAttrib( "vNrm", "3,GL_FLOAT,GL_FALSE,12,0,false" ); 

    // NB: careful with the ordering of the above or the OpenGL state machine will bite you : 
    // the vPos and vNrm attribs needs to ne enabled after the appropriate buffer is made THE active GL_ARRAY_BUFFER

    int num = ssys::getenvint("NUM",mesh->indices_num) ; 
    int off = ssys::getenvint("OFF",mesh->indices_offset) ; 

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();  // calls gl.updateMVP
        glDrawElements(GL_TRIANGLES, num, GL_UNSIGNED_INT, (GLvoid*)(sizeof(GLuint) * off ));
        gl.renderloop_tail(); 
    }
    exit(EXIT_SUCCESS);
}

