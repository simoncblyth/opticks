Clipping Mode With GDML/glTF geometry make instances disappear
================================================================


Issue
---------

Very obvious : just press C  and all PMTs disappear, and some other instanced
geometry disappears too.



FIXED
-------

By applying InstanceTransform prior to clipping in oglrap/gl/inrm/vert.glsl 


* NB shader modifications despite being "compiled" at run time 
  still need to be installed via the standard CMake build "oglrap--" 
  

OpenGL gl_ClipDistance
--------------------------

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_ClipDistance.xhtml




Clipping in oglrap and okc
-------------------------------


::

    400 void Interactor::key_pressed(unsigned int key)
    401 {
    409     switch (key)
    410     {
    ...
    417         case GLFW_KEY_B:
    418             m_scene->nextGeometryStyle();
    419             break;
    420         case GLFW_KEY_C:
    421             m_clipper->next();
    422             break;


::

    simon:oglrap blyth$ grep Clip *.*
    GUI.cc:#include "Clipper.hh"
    GUI.cc:    setClipper(composition->getClipper());
    GUI.cc:void GUI::setClipper(Clipper* clipper)
    GUI.cc:void GUI::clipper_gui(Clipper* clipper)
    GUI.cc:    if (ImGui::CollapsingHeader("Clipper"))
    GUI.hh:class Clipper ;
    GUI.hh:       void setClipper(Clipper* clipper);
    GUI.hh:       void clipper_gui(Clipper* clipper);
    GUI.hh:       Clipper*      m_clipper ; 
    Interactor.cc:#include "Clipper.hh"
    Interactor.cc:    m_clipper  = composition->getClipper();
    Interactor.cc:"\n C: Clipper::next             toggle geometry clipping "
    Interactor.hh:class Clipper ;
    Interactor.hh:       Clipper*     m_clipper ; 
    Rdr.cc:        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());
    Rdr.cc:        glUniformMatrix4fv(m_isnorm_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipISNormPtr());
    Renderer.cc:        m_clip_location = m_shader->uniform("ClipPlane",          required); 
    Renderer.cc:        m_clip_location = m_shader->uniform("ClipPlane",          required); 
    Renderer.cc:        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());
    Renderer.cc:        glUniform4fv(m_clip_location, 1, m_composition->getClipPlanePtr() );
    Renderer.cc:        if(m_composition->getClipMode() == -1)
    Renderer.cc:            print( m_composition->getClipPlanePtr(), "Renderer::update_uniforms ClipPlane", 4);
    Renderer.cc:                glm::mat4 w2c = glm::make_mat4(m_composition->getWorld2ClipPtr()); 
    oglrap.bash:       GLFW event handling and passing off to Camera, Trackball, View, Clipper etc..
    oglrap.bash:       matrix calculations based on the Camera, Trackball, View and Clipper constituents
    oglrap.bash:Clipper
    oglrap.bash:ClipperCfg
    simon:oglrap blyth$ 

::

    494 void Renderer::update_uniforms()
    ...
    519         glUniform4fv(m_lightposition_location, 1, m_composition->getLightPositionPtr());
    520 
    521         glUniform4fv(m_clip_location, 1, m_composition->getClipPlanePtr() );
    522 
    524         glm::vec4 cd = m_composition->getColorDomain();
    525         glUniform4f(m_colordomain_location, cd.x, cd.y, cd.z, cd.w  );
    ...
    534         if(m_composition->getClipMode() == -1)
    535         {
    536             glDisable(GL_CLIP_DISTANCE0);
    537         }   
    538         else
    539         {
    540             glEnable(GL_CLIP_DISTANCE0);
    541         }   


oglrap/gl/nrm/vert.glsl::

     16 layout(location = 0) in vec3 vertex_position;
     17 layout(location = 1) in vec3 vertex_colour;
     18 layout(location = 2) in vec3 vertex_normal;
     19         
     20 float gl_ClipDistance[1];
     21     
     22 out vec4 colour;
     23         
     24 void main () 
     25 {   
     26     //
     27     // NB using flipped normal, for lighting from inside geometry
     28     //  
     29     //    normals are expected to be outwards so the natural
     30     //    sign of costheta is negative when the light is inside geometry
     31     //    thus in order to see something flip the normals
     32     //
     33 
     34     float flip = NrmParam.x == 1 ? -1. : 1. ;
     35 
     36     vec3 normal = flip * normalize(vec3( ModelView * vec4 (vertex_normal, 0.0)));
     37 
     38     vec3 vpos_e = vec3( ModelView * vec4 (vertex_position, 1.0));  // vertex position in eye space 
     39 
     40     gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);
     41 
     42     vec3 ambient = vec3(0.1, 0.1, 0.1) ;
     43 
     44 #incl vcolor.h
     45 
     46     gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);
     47 
     48 }


oglrap/gl/inrm/vert.glsl::

     22 void main ()
     23 {
     24     //
     25     // NB using flipped normal, for lighting from inside geometry
     26     //
     27     //    normals are expected to be outwards so the natural
     28     //    sign of costheta is negative when the light is inside geometry
     29     //    thus in order to see something flip the normals
     30     //
     31 
     32     float flip = NrmParam.x == 1 ? -1. : 1. ;
     33 
     34     vec3 normal = flip * normalize(vec3( ModelView * vec4 (vertex_normal, 0.0)));
     35 
     36     vec3 vpos_e = vec3( ModelView * InstanceTransform * vec4 (vertex_position, 1.0));  // vertex position in eye space
     37 
     38     gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          FIX BY APPLYING InstanceTransform  BEFORE SETTING ClipDistance
     39 
     40     vec3 ambient = vec3(0.1, 0.1, 0.1) ;
     41 
     42 #incl vcolor.h
     43 
     44     gl_Position = ModelViewProjection * InstanceTransform * vec4 (vertex_position, 1.0);
     45 
     46 }
     47 







