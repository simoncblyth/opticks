j1707 : OpenGL LOD (Level of detail)
=======================================

* :google:`opengl instanced rendering with level of detail`

* some instances will be close and some far, 


Investigations
------------------

See::

    env- ; instcull- ; icdemo 


Unity 
-------

* https://docs.unity3d.com/Manual/LevelOfDetail.html
* https://docs.unity3d.com/Manual/GPUInstancing.html


Culling
---------

* http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/

The applicability of geometry instancing is strongly limited by several
factors. One of the most important ones is the culling of instanced geometries.
One may choose to cull these objects in the same fashion as others, using the
CPU, but that usually breaks the batch and maybe we loose the benefits of
geometry instancing. It is more and more imminent to have a GPU based
alternative. Without CPU based culling, by sending the whole bunch of instances
down the graphics pipeline may choke our vertex processor in case we have high
poly geometries and quite large amount of instances of it.

The rendering technique presented in this article will try to achieve this
goal. We will use a multi-pass technique that in the first pass culls the
object instances against the view frustum using the GPU and in the second pass
renders only those instances that are likely to be visible in the final scene.
This way we can severely reduce the amount of vertex data sent through the
graphics pipeline.


GPU based dynamic geometry LOD
-----------------------------------

* http://rastergrid.com/blog/2010/10/opengl-4-0-mountains-demo-released/
* http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/


GPU based dynamic geometry LOD determination using a geometry shader that
selects the most appropriate LOD from a group of geometry LODs based on the
object’s distance from camera.

*  LOD determination pass can be also merged together with other visibility determination passes 

* OpenGL 4.0 (see the extension ARB_transform_feedback3) 



Fork vertices into three streams, depending on vertex distance.

* hmm so you push 3 LOD meshes thru that ? call that 

::

    #version 400 core

    uniform mat4 ModelViewMatrix;
    uniform vec2 LodDistance;

    layout(points) in;
    layout(points, max_vertices = 1) out;

    in vec3 InstancePosition[1];

    layout(stream=0) out vec3 InstPosLOD0;
    layout(stream=1) out vec3 InstPosLOD1;
    layout(stream=2) out vec3 InstPosLOD2;

    void main() {
      float distance = length(ModelViewMatrix * vec4(InstancePosition[0], 1.0));
      if ( distance < LodDistance.x ) {
        InstPosLOD0 = InstancePosition[0];
        EmitStreamVertex(0);
      } else
      if ( distance < LodDistance.y ) {
        InstPosLOD1 = InstancePosition[0];
        EmitStreamVertex(1);
      } else {
        InstPosLOD2 = InstancePosition[0];
        EmitStreamVertex(2);
      }
    }



EmitStreamVertex
-----------------

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/EmitStreamVertex.xhtml

Available from OpenGL 4.0::

   void EmitStreamVertex(   int stream);    emit a vertex to a specified stream

   Available only in the Geometry Shader, EmitStreamVertex emits the current
   values of output variables to the current output primitive on stream stream.
   The argument stream must be a constant integral expression. On return from this
   call, the value of all output variables for all streams are undefined.



Geometry_Shader Transform Feedback
-------------------------------------

* https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_transform_feedback3.txt
* https://www.khronos.org/opengl/wiki/Geometry_Shader

When using Transform Feedback to compute values, it is often useful to be able
to send different sets of vertices to different buffers at different rates. For
example, GS's can send vertex data to one stream, while building per-instance
data in another stream. The vertex data and per-instance data will be of
different lengths, written at different speeds.

Multiple stream output requires that the output primitive type be points. You
can still take whatever input you prefer.

To provide this, output variables can be given a stream index with a layout
qualifier.




(From OpenGL 4.0) glBeginQueryIndexed, glEndQueryIndexed 
---------------------------------------------------------

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBeginQueryIndexed.xhtml

delimit the boundaries of a query object on an indexed target

::

    void glBeginQueryIndexed(    
        GLenum target,
        GLuint index,
        GLuint id);
     
    void glEndQueryIndexed( 
        GLenum target,
        GLuint index);

    target 

        Specifies the target type of query object established between
        glBeginQueryIndexed and the subsequent glEndQueryIndexed. 

        The symbolic constant must be one of 

        * GL_SAMPLES_PASSED
        * GL_ANY_SAMPLES_PASSED,
        * GL_PRIMITIVES_GENERATED
        * GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN
        * GL_TIME_ELAPSED.    

    index
        Specifies the index of the query target upon which to begin the query.

    id
        Specifies the name of a query object.


   
GL_PRIMITIVES_GENERATED
~~~~~~~~~~~~~~~~~~~~~~~~

If target is GL_PRIMITIVES_GENERATED, id must be an unused name, or the name of
an existing primitive query object previously bound to the
GL_PRIMITIVES_GENERATED query binding. 

When glBeginQueryIndexed is executed, the query object's primitives-generated 
counter is reset to 0. Subsequent rendering will increment the counter once 
for every vertex that is emitted from the geometry shader to the stream 
given by index, or from the vertex shader if index is zero and no geometry shader is present. 
When glEndQueryIndexed is executed, the primitives-generated counter for stream index 
is assigned to the query object's result value. This value can be queried by calling
glGetQueryObject with pname GL_QUERY_RESULT. When target is GL_PRIMITIVES_GENERATED, 
index must be less than the value of GL_MAX_VERTEX_STREAMS.



::

    for (int i=0; i<NUM_LOD; i++)
      glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, lodQuery[i]);

    glBeginTransformFeedback(GL_POINTS);
      glDrawArrays(GL_POINTS, 0, instanceCount);
    glEndTransformFeedback();

    for (int i=0; i<NUM_LOD; i++)
      glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i);



::

    for (int i=0; i<NUM_LOD; i++) 
    {
      glGetQueryObjectiv(lodQuery[i], GL_QUERY_RESULT, instanceCountLOD[i]);
      if ( instanceCountLOD[i] > 0 )
        glDrawElementsInstanced(..., instanceCountLOD[i]);
    }





oglrap instanced
------------------


::

    575 void Renderer::render()
    576 {
    577     glUseProgram(m_program);
    578 
    579     update_uniforms();
    580 
    581     bind();
    582 
    583     // https://www.opengl.org/archives/resources/faq/technical/transparency.htm
    584     glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    585     glEnable (GL_BLEND);
    586 
    587     if(m_wireframe)
    588     {
    589         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    590     }
    591 
    592     if(m_instanced)
    593     {
    594         // primcount : Specifies the number of instances of the specified range of indices to be rendered.
    595         //             ie repeat sending the same set of vertices down the pipeline
    596         //
    597         GLsizei primcount = m_itransform_count ;
    598         glDrawElementsInstanced( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL, primcount  ) ;
    599     }
    600     else
    601     {
    602         glDrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL ) ;
    603     }
    604     // indices_count would be 3 for a single triangle, 30 for ten triangles
    605 
    606 
    607     //
    608     // TODO: try offsetting into the indices buffer using : (void*)(offset * sizeof(GLuint))
    609     //       eg to allow wireframing for selected volumes
    610     //
    611     //       need number of faces for every volume, so can cumsum*3 to get the indice offsets and counts 
    612     //
    613     //       http://stackoverflow.com/questions/9431923/using-an-offset-with-vbos-in-opengl
    614     //
    615 
    616     if(m_wireframe)
    617     {
    618         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    619     }
    620 
    621 
    622     m_draw_count += 1 ;
    623 
    624     glBindVertexArray(0);
    625 
    626     glUseProgram(0);
    627 }

