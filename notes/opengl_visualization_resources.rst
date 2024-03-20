opengl_visualization_resources
================================


OpenGL introductions
---------------------

Note that mathematics of view, projection etc.. is 
common to all graphics libraries. 
So studying OpenGL learning resources can help with any rendering 
not just OpenGL. 

* https://learnopengl.com/Introduction
* http://www.opengl-tutorial.org/


* https://nicolbolas.github.io/oldtut/index.html

* https://paroj.github.io/gltut/index.html
* https://paroj.github.io/gltut/Basics/Tut01%20Following%20the%20Data.html


For arranging consistent ray trace and rasterized renders
-----------------------------------------------------------

* https://antongerdelan.net/opengl/raycasting.html


Debugging
----------

* https://learnopengl.com/In-Practice/Debugging


OpenGL expert on Stackoverflow
---------------------------------

* https://stackoverflow.com/users/734069/nicol-bolas



* https://nicolbolas.github.io/oldtut/index.html

* https://nicolbolas.github.io/oldtut/Basics/Intro%20What%20is%20OpenGL.html

* https://nicolbolas.github.io/oldtut/Basics/Tut01%20Dissecting%20Display.html



Rendering Multiple Meshes
---------------------------

* https://www.reddit.com/r/opengl/comments/n9eagu/how_to_render_multiple_meshes_in_opengl/

Depending on the size of your models, you may be able to package all of them
into a single VBO and a single IBO. Then you can format a draw buffer and use
glMultiDrawElementsIndirect to render every single object with a single draw
call. That would probably be the most efficient way to do this.

But simply instancing all objects of the same model together, thus using 3-8
draw calls will probably be close to the same real world performance and
requires less work to set up :



* https://www.spacesimulator.net/tutorials/OpenGL_matrices_tutorial_3_3.html


glMultiDrawElementsIndirect (from OpenGL 4.3?)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://registry.khronos.org/OpenGL-Refpages/gl4/html/glMultiDrawElementsIndirect.xhtml


Table has tick for 4.3 but also note stating 4.2 works partially::

    The baseInstance member of the DrawElementsIndirectCommand structure is defined
    only if the GL version is 4.2 or greater. For versions of the GL less than 4.2,
    this parameter is present but is reserved and should be set to zero. On earlier
    versions of the GL, behavior is undefined if it is non-zero. 


* old laptop GPU only handles 4.1 
* BUT: OpenGL impls are slippery, so just try 

Equiv to drawcount loop calling glDrawElementsInstancedBaseVertexBaseInstance

* could detect the OpenGL version and spell out the loop for old laptop 


glDrawElementsInstancedBaseVertexBaseInstance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://registry.khronos.org/OpenGL-Refpages/gl4/html/glDrawElementsInstancedBaseVertexBaseInstance.xhtml


glDrawElementsIndirect
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://registry.khronos.org/OpenGL-Refpages/gl4/html/glDrawElementsIndirect.xhtml



Instancing
------------

* https://learnopengl.com/Advanced-OpenGL/Instancing


* Using uniforms and gl_InstanceID 
  does not scale beyond a few hundred

* need to shoehorn 4x4 instance transforms into vertex arrays



* ~/env/graphics/opengl/instance/instance.cc
* ~/opticks/examples/UseInstance

* ~/opticks/oglrap/gl/inrm/vert.glsl 



oglrap/Renderer::

     469 GLuint Renderer::createVertexArray(RBuf* instanceBuffer)
     470 {
     ...
     499     GLboolean normalized = GL_FALSE ;
     500     GLsizei stride = 0 ;
     501     const GLvoid* offset = NULL ;
     502 
     503     if(instanceBO > 0)
     504     {
     505         LOG(verbose) << "Renderer::upload_buffers setup instance transform attributes " ;
     506         glBindBuffer (GL_ARRAY_BUFFER, instanceBO);
     507 
     508         uintptr_t qsize = sizeof(GLfloat) * 4 ;
     509         GLsizei matrix_stride = qsize * 4 ;
     510 
     511         glVertexAttribPointer(vTransform + 0 , 4, GL_FLOAT, normalized, matrix_stride, (void*)0 );
     512         glVertexAttribPointer(vTransform + 1 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize));
     513         glVertexAttribPointer(vTransform + 2 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*2));
     514         glVertexAttribPointer(vTransform + 3 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*3));
     515 
     516         glEnableVertexAttribArray (vTransform + 0);
     517         glEnableVertexAttribArray (vTransform + 1);
     518         glEnableVertexAttribArray (vTransform + 2);
     519         glEnableVertexAttribArray (vTransform + 3);
     520 
     521         GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on to     p of each other
     522         glVertexAttribDivisor(vTransform + 0, divisor);  // dictates instanced geometry shifts between instances
     523         glVertexAttribDivisor(vTransform + 1, divisor);
     524         glVertexAttribDivisor(vTransform + 2, divisor);
     525         glVertexAttribDivisor(vTransform + 3, divisor);
     526     }



~/opticks/oglrap/gl/inrm/vert.glsl::

     24 uniform mat4 ModelViewProjection ;
     25 uniform mat4 ModelView ;
     26 uniform vec4 ClipPlane ;
     27 uniform vec4 LightPosition ; 
     28 uniform vec4 Param ;
     29 uniform ivec4 NrmParam ;
     30 
     31 
     32 layout(location = 0) in vec3 vertex_position;
     33 layout(location = 1) in vec3 vertex_colour;
     34 layout(location = 2) in vec3 vertex_normal;
     35 layout(location = 4) in mat4 InstanceTransform ;
     36 
        

     41 void main () 
     42 {
     ..
     51     float flip = NrmParam.x == 1 ? -1. : 1. ;
     52 
     53     vec3 normal = flip * normalize(vec3( ModelView * vec4 (vertex_normal, 0.0)));
     54 
     55 
     56     vec4 i_vertex_position = InstanceTransform * vec4 (vertex_position, 1.0) ;
     57 
     58 
     59     vec3 vpos_e = vec3( ModelView * i_vertex_position);  // vertex position in eye space
     60 
     61     gl_ClipDistance[0] = dot(i_vertex_position, ClipPlane);
     62 
     63     vec3 ambient = vec3(0.1, 0.1, 0.1) ;
     64 
     65 #incl vcolor.h
     66 
     67     gl_Position = ModelViewProjection * i_vertex_position ;
     68 
     69 }


Compositing
-------------

* :google:`OpenGL Compositing ray trace and rasterized`



ancient laptop OpenGL 4.1 issue
-----------------------------------


::

    122         GLint basevertex = 0 ;
    123         GLuint baseinstance = 0 ;
    124         glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance );
    125         // SEGV on laptop, OK on worktation 
    126         // https://github.com/moderngl/moderngl/issues/346
    127         std::cout << "SGLFW_Render::render_drawElements.glDrawElementsInstancedBaseVertexBaseInstance" << std::endl ;
    128     




