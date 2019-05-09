geometry-shaders-are-deprecated-how-to-replace
================================================




* :google:`opengl replace geometry shader with compute shader`


Constraints on replacement for geometry shaders
------------------------------------------------


* dont want to require CUDA to visualize propagations : want to be able to do it all in OpenGL  



Talk 2016 : GPU driven rendering  
----------------------------------

* http://on-demand.gputechconf.com/gtc/2016/presentation/s6138-christoph-kubisch-pierre-boudier-gpu-driven-rendering.pdf

Present: Modern APIs and data-driven design methods provide efficient ways to
render scenes OpenGL Multi Draw Indirect, NVIDIA bindless and
NV_command_list technology

Vulkan native support for command-buffers and general design that 
allows re-use and much better control over validation costs

Passthrough geometry shader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.khronos.org/registry/OpenGL/extensions/NV/NV_geometry_shader_passthrough.txt


Use "shuffle" in vertex shader to access other vertices of same primitive ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* googling gives no corroboration : must be "typo" OR very new
* https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt
* https://developer.nvidia.com/reading-between-threads-shader-intrinsics


pipeline book
~~~~~~~~~~~~~~~~

* https://simonschreibt.de/gat/renderhell/


cudaraster (Laine and Karras)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://research.nvidia.com/publication/high-performance-software-rasterization-gpus

Efficient Buffer Management (John McDonald 2012)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.gamedevs.org/uploads/efficient-buffer-management.pdf

* Alignment matters! (16-byte, please) : Aligned copies can be ~30x faster
* One bad sync point can halve your frame rate
* Generally, the later the sync point, the worse it is


NVIDIA : OpenGL Graphics and Compute Samples
-----------------------------------------------

* https://docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/gl-samples.htm


Gameworks Samples
~~~~~~~~~~~~~~~~~~~

* see env- gameworks- 

* https://developer.nvidia.com/gameworks-samples-overview

* https://github.com/NVIDIAGameWorks/GraphicsSamples

* https://docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/feedbackparticlessample.htm

The Feedback Particles sample shows how normal vertex shaders can be used to
animate particles and write the results back into vertex buffer objects via
Transform Feedback, for use in subsequent frames. This is another way of
implementing GPU-only particle animations. The sample also uses Geometry
Shaders to generate custom particles from single points and also to kill the
dead ones.

::

   interesting idea : remember that the rendering code is being called 30-60 
   times a second : so avoid doubling up buffers : it doesnt matter if the first 
   few frames are incorrect : also remember OpenGL buffer cycling, so only 
   update the next frame to be displayed (not the current) 




Designworks Samples
~~~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/samples

  * https://github.com/nvpro-samples

  * https://libcinder.org/




Compute Shaders : OpenGL core since 4.3 (mid 2012)
------------------------------------------------------

* https://www.khronos.org/opengl/wiki/Compute_Shader

* http://antongerdelan.net/opengl/compute.html

  Example that uses compute shader to write into a texture


Using Compute Shader to write to buffer
-----------------------------------------

* https://computergraphics.stackexchange.com/questions/6045/how-to-use-the-data-manipulated-in-opengl-compute-shader

gallickgunner:

I think you are confusing all the binding targets thingy. From what I see your
vertex data is coming from compute shader after some processing and now you
want to pass it to the vertex shader.

You can create a buffer object once, use it as an SSBO for use in a compute
shader then use it as a VBO for use in rendering. i.e

::

    // Setup SSBO for use in Compute shader
    glGenBuffers(1, &vbo_Current);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbo_Current);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, &vertices[0], GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo_Current);

    // Setup the same Buffer object as VBO for use in rendering
    glBindBuffer(GL_ARRAY_BUFFER, vbo_Current);   <- Note the change in binding target
    GlVertexAttribPointer().......                <- set your vertex data

Hence there is no need to generate 1 more buffer object for the texture, just
pass the same buffer but change the binding points to GL_ARRAY_BUFFER

However do note that you need to use appropriate Memory barrier calls as SSBO
reads and writes are incoherent memory accesses. Don't know much about these
but you can find more information here


SSBO : GL_SHADER_STORAGE_BUFFER
----------------------------------

* https://community.arm.com/developer/tools-software/graphics/b/blog/posts/get-started-with-compute-shaders

The Shader Storage Buffer Object (SSBO) feature for instance has been
introduced along with compute shaders and that gives additional possibilities
for exchanging data between pipeline stages, as well as being flexible input
and output for compute shaders.

* https://stackoverflow.com/questions/42062621/gl-shader-storage-buffer-memory-limitations

The minimum maximum size of a shader storage buffer is represented by the
symbolic constant MAX_SHADER_STORAGE_BLOCK_SIZE as per section 7.8 of the core
OpenGL 4.5 specification.

Since their adoption into core, the required size (i.e. the minimum maximum)
has been significantly increased. In core OpenGL 4.3 and 4.4, the minimum
maximum was pow(2, 24) (or 16MB with 1 byte basic machine units and 1MB =
1024^2 bytes) - in core OpenGL 4.5 this value is now pow(2, 27) (or 128MB)

* https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf
* ~/opticks_refs/OpenGL_45_Core_Specificaton_2017_glspec45.core.pdf


* https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object

The spec guarantees that SSBOs can be up to 128MB. Most implementations will
let you allocate a size up to the limit of GPU memory.


* https://www.geeks3d.com/20140704/tutorial-introduction-to-opengl-4-3-shader-storage-buffers-objects-ssbo-demo/

* https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_storage_buffer_object.txt


Size of Opticks record buffer
--------------------------------

The records buffer of has shape (3M, 16, 2, 4) with each step point 
domain compressed into 2*4 shorts (16 bits) totalling 128 bits. 

::

    3M * 16 * 128 bits

    3M * 16 * 16 bytes   : 768MB for 3M photons  





Opticks record buffer Geometry Shaders all follow a similar pattern oglrap/gl/rec/geom.glsl
--------------------------------------------------------------------------------------------------

* used to render the highly compressed record buffer of the photon propagation, which 
  stores positions and times at up to 16 points of a propagation 

Critical capabilities of geometry shaders used:

* discard primitives if invalid time combinations based in uniform input time

  * vertex shader cannot do this, but it can set a flag to cause discard in fragment shader

* also it interpolates between two input positions of "a line" based on time input 
  requiring access to two vertices at once

  * vertex shader does not have that capability?



::

     01 #version 410 core
      2 //  rec/geom.glsl : flying point
      3 
      4 #incl dynamic.h
      5 
      6 uniform mat4 ISNormModelViewProjection ;
      7 uniform vec4 TimeDomain ;
      8 uniform vec4 ColorDomain ;
      9 uniform vec4 Param ;
     10 
     11 uniform  vec4 ScanParam ;
     12 uniform ivec4 NrmParam ;
     13 
     14 //more efficient to skip in geometry shader rather than in fragment, if possible
     15 
     16 
     17 uniform ivec4 Pick ;
     18 uniform ivec4 RecSelect ;
     19 uniform ivec4 ColorParam ;
     20 uniform ivec4 PickPhoton ;
     21 
     22 uniform sampler1D Colors ;
     23 
     24 in uvec4 flq[];
     25 in ivec4 sel[];
     26 in vec4 polarization[];
     27 
     28 layout (lines) in;
     29 layout (points, max_vertices = 1) out;
     30 
     31 out vec4 fcolor ;
     32 
     33 void main ()
     34 {
     35     uint seqhis = sel[0].x ;
     36     uint seqmat = sel[0].y ;
     37     if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
     38     if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;
     39 
     40     uint photon_id = gl_PrimitiveIDIn/MAXREC ;
     41     if( PickPhoton.x > 0 && PickPhoton.y > 0 && PickPhoton.x != photon_id )  return ;
     42 
     43 
     44     vec4 p0 = gl_in[0].gl_Position  ;
     45     vec4 p1 = gl_in[1].gl_Position  ;
     46     float tc = Param.w / TimeDomain.y ;
     47 
     48     uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ;
     49     uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;
     50     uint vselect = valid & select ;
     51 
     52 #incl fcolor.h
     53 
     54     if(vselect == 0x7) // both valid and straddling tc
     55     {
     56         vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
     57         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     58 
     59         if(NrmParam.z == 1)
     60         {
     61             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     62             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     63         }
     64 
     65 
     66         EmitVertex();
     67         EndPrimitive();
     68     }
     69     else if( valid == 0x7 && select == 0x5 )     // both valid and prior to tc
     70     {
     71         vec3 pt = vec3(p1) ;
     72         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     73 
     74         if(NrmParam.z == 1)
     75         {
     76             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     77             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     78         }
     79 
     80 
     81         EmitVertex();
     82         EndPrimitive();
     83     }
     84 
     85 }




Geometry Shaders cause serialization...
------------------------------------------


* http://www.joshbarczak.com/blog/?p=667

  Why Geometry Shaders Are Slow 


Wolfgang Engel March 18, 2015 at 6:18 pm

That was a good read. The geometry shader is writing into memory on some
platforms … one of the work arounds is to try to re-write every geometry shader
based algorithm as vertex shader instancing.

The general advice for many platforms is that one doesn’t want to use the
geometry shader at all since a few years. This changed over time with different
hardware architectures. As far as I can tell the three hardware vendors you
mention didn’t have any consistent support of the geometry shader over the
years. It changed. The trend is going away from the geometry shader usage ….

On a related note: we should just drop all the different shader stages and use
instead an extended compute shader that exposes all the functionality … and
just does a couple of runs …




* https://stackoverflow.com/questions/50557224/metal-emulate-geometry-shaders-using-compute-shaders



* https://gamedev.stackexchange.com/questions/48432/why-does-this-geometry-shader-slow-down-my-program-so-much



