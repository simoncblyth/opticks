
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
 

rec

altrec

devrec



Unpartitioned Record structure 
----------------------------------

Below ascii art shows expected pattern of slots and times for MAXREC 5 
    
* remember that from point of view of shader the input time is **CONSTANT**
  think of the drawing as a chart plotter tracing over all the steps of all the photons, 
  this shader determines when to put the pen down onto the paper
     
  * it needs to lift pen between photons and avoid invalids 
    
  * slot indices are presented modulo 5
  * negative times indicates unset
  * dt < 0. indicates p1 invalid

::

    //  
    //
    //     |                                          
    //     |                                           
    //     t                                            
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


