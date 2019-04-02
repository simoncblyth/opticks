OpenGL Render Pipelines
=========================

Geometry 
----------

nrm
     steered by *Renderer*

tex
     used for OptiX raycast renders, OptiX populates a texture that OpenGL presents


Gensteps
--------------- 

p2l
     point to line geometry shader used to render gensteps


Photons
--------------

pos
     point representation of *photon* positions
     (not a geometry shader)


Records : corresponding to each recorded step of the photon
-------------------------------------------------------------

The records buffer of has shape (3M, 16, 2, 4) with each step point 
domain compressed into 2*4 shorts (16 bits) totalling 128 bits. 

Relevant sources:

* *oxrap/cu/generate.cu* 
* *oxrap/cu/photon.h

The visualization renders this buffer with a single *glDrawArrays* in 
*oglrap/Rdr.cc:render*  which uses an OpenGL geometry shader with the 
event time as an input uniform.
Actually there are three variants of the record renderer, presenting the photons
as a flying point, shortline or longline.

 
rec
      flying point presentation, oglrap/Scene.cc:m_record_renderer
altrec
      long line strip presentation, oglrap/Scene.cc:m_altrecord_renderer
devrec
      vector (short line) presentation, vector length is controllable interactively 
      via Composition/param.y, oglrap/Scene.cc:m_devrecord_renderer 


The geometry shader is the crucial thing that must be understood to see how 
the visualization works:

oglrap/gl/rec/geom.glsl
    flying point  
oglrap/gl/devrec/geom.glsl
    shortline "vector" 
oglrap/gl/altrec/geom.glsl 
    longline 


The input primitives to all three renderers are the same, LINE_STRIP, 
but the output primitive is  

::

     401 void Scene::initRenderers()
     402 {
     ...
     470     m_record_renderer = new Rdr(m_device, "rec", m_shader_dir, m_shader_incl_path );
     471     m_record_renderer->setPrimitive(Rdr::LINE_STRIP);
     472 
     473     m_altrecord_renderer = new Rdr(m_device, "altrec", m_shader_dir, m_shader_incl_path);
     474     m_altrecord_renderer->setPrimitive(Rdr::LINE_STRIP);
     475 
     476     m_devrecord_renderer = new Rdr(m_device, "devrec", m_shader_dir, m_shader_incl_path);
     477     m_devrecord_renderer->setPrimitive(Rdr::LINE_STRIP);



Geometry Shader Docs
-----------------------

The defining feature of the geometry shader is its ability to 
amplify or reduce geometry with respect to the input primitives.  
The Opticks photon visualization relies on the geometry shaders 
ability to emit zero primitives depending on the values in the 
record buffer that identify non-valid    



* https://www.khronos.org/opengl/wiki/Geometry_Shader

=========  ====================  ================
GS input    OpenGL primitives     vertex count
=========  ====================  ================
points      GL_POINTS              1
lines       GL_LINES,              2
            GL_LINE_STRIP, 
            GL_LINE_LIST      
=========  ====================  ================


The output_primitive must be one of the following:

* points
* line_strip
* triangle_strip

These work exactly the same way their counterpart OpenGL rendering modes do. To
output individual triangles or lines, simply use EndPrimitive (see below) after
emitting each set of 3 or 2 vertices.


Record Renderer Geometry Shaders
----------------------------------

oglrap/rec/geom.glsl::
     28 layout (lines) in;
     29 layout (points, max_vertices = 1) out;

oglrap/devrec/geom.glsl::
     23 layout (lines) in;
     24 layout (line_strip, max_vertices = 2) out;

oglrap/altrec/geom.glsl::
     22 layout (lines) in;
     23 layout (line_strip, max_vertices = 2) out; 


Unpartitioned Record structure 
----------------------------------

Below ascii art shows the pattern of record buffer slots and times 
for MAXREC 5 (for ease of presentation, actually MAXREC of 16 or 10 is used).
    
* remember that from the point of view of the shader the input time is **CONSTANT**
  think of the drawing as a chart plotter tracing over all the steps of all the photons, 
  this shader determines when to put the pen down onto the paper
     
  * it needs to lift pen between photons and avoid invalids 
    
  * slot indices are presented modulo 5
  * negative times indicates unset
  * dt < 0. indicates p1 invalid

::

    //     t
    //
    //     |                                          
    //     |                                           
    //     |                                            
    //     |          3                                  
    //     |                                          4
    //     |      2                                3
    //     |    1                               2              
    //     |  0                   2          1              1 
    //     |                    1         0               0          0
    //     +-----------------0--------> slot ------------------------------------->
    //     |                                     
    //     |              4         3 4                        2 3 4    1 2 3 4 
    //     |
    //
    //   
     
* geom shader gets to see all consequtive pairs 
  (including invalid pairs that cross between different photons)
    
* shader uses one input time cut Param.w to provide history scrubbing 
    
* a pair of contiguous recs corresponding to a potential line
      
Choices over what to do with the pair:
    
* do nothing with this pair, eg for invalids 
* interpolate the positions to find an intermediate position 
  as a function of input time 
    
* throw away one position, retaining the other 
      
* https://www.opengl.org/wiki/Geometry_Shader
* http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2


    
Cannot form a line with only one valid point ? unless conjure a constant direction.
The only hope is that a prior "thread" got the valid point as
the second of a pair. 

Perhaps that means must draw with GL_LINE_STRIP rather than GL_LINES in order
that the geometry shader sees each vertex twice (?)   YES : SEEMS SO
      
Hmm how to select single photons/steps ?  
     
* Storing photon identifies occupies ~22 bits at least (1 << 22)/1e6 ~ 4.19
* Step identifiers 
   
* https://www.opengl.org/wiki/Built-in_Variable_(GLSL) 
    
* https://www.opengl.org/sdk/docs/man/html/gl_VertexID.xhtml
   
  non-indexed: it is the effective index of the current vertex (number of vertices processed + *first* value)
  indexed:   index used to fetch this vertex from the buffer
    
  * control the the glDrawArrays first/count to pick the desired range  
  * adopt glDrawElements and control the indices
    

Geometry Shader Background

* https://www.opengl.org/wiki/Geometry_Shader
* http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2


