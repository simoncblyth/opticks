# === func-gen- : graphics/oglrap/oglrap fgp graphics/oglrap/oglrap.bash fgn oglrap fgh graphics/oglrap
oglrap-src(){      echo graphics/oglrap/oglrap.bash ; }
oglrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglrap-src)} ; }
oglrap-vi(){       vi $(oglrap-source) ; }
oglrap-env(){      elocal- ; }
oglrap-usage(){ cat << EOU

Featherweight OpenGL wrapper
==============================

Just a few utility classes to make modern OpenGL 3, 4 
easier to use.

Originally was thinking could keep GLFW3 out of oglrap
keeping pure OpenGL, but that has proved difficult.

GLFW3 and GLEW are responsible for furnishing the 
OpenGL headers, but wanted to avoid directly using 
GLFW3 inside oglrap-


Better Shader Handling ?
--------------------------

* want less tedium when adding/setting uniforms/attributes 
 
  * https://github.com/mmmovania/opengl33_dev_cookbook_2013/blob/master/Chapter2/src/GLSLShader.cpp

  * https://github.com/OpenGLInsights/OpenGLInsightsCode


TODO
-----

* add clipping planes to "nrm" shaders as check of 
  shader uniform handling and as need to clip 


Classes
--------


Frame
       OpenGL context creation and window control 
Interactor
       GLFW event handling and passing off to Camera, Trackball, View etc..

Composition
       matrix manipulations based on the Camera, Trackball and View constituents
Camera
       near/far/...
Trackball
       quaternion calulation of perturbing rotations, and translations too
View  
       eye/look/up  

Geometry
      high level control of geometry loading 

Rdr
       specialization of RendererBase currently used for 
       the below tags (ie sets of GLSL programs )

       pos : used directly from GGeoView main for 
             visualizing VecNPY event data
Renderer 
       specialization of RendererBase currently used 
       the below tags (ie sets of GLSL programs )
 
       nrm : normal shader used directly from GGeoView main
       tex : quad texture used by OptiXEngine   

       /// will continue with 2 renderers for a while
       /// until experience dictates which approach is best 
RendererBase
       handles shader program access, compilation and linking 
       using Prog and Shdr classes
Prog 
       representation of shader program pipeline 
       comprised of one for more shaders
Shdr
       single shader


Texture
      misnamed : QuadTex might be better
      Used for rendering OptiX generated PBOs Pixel Buffer Objects 
      via OpenGL textures 

Demo
      GMesh subclass representing a single triangle geometry  


CameraCfg
CompositionCfg
FrameCfg
InteractorCfg
RendererCfg
TrackBallCfg
ViewCfg
      configuration connector classes enabling commandline or live 
      config of objects





EOU
}


oglrap-sdir(){ echo $(env-home)/graphics/oglrap ; }
oglrap-idir(){ echo $(local-base)/env/graphics/oglrap ; }
oglrap-bdir(){ echo $(oglrap-idir).build ; }

oglrap-bindir(){ echo $(oglrap-idir)/bin ; }

oglrap-scd(){  cd $(oglrap-sdir); }
oglrap-cd(){   cd $(oglrap-sdir); }

oglrap-icd(){  cd $(oglrap-idir); }
oglrap-bcd(){  cd $(oglrap-bdir); }
oglrap-name(){ echo OGLRap ; }

oglrap-wipe(){
   local bdir=$(oglrap-bdir)
   rm -rf $bdir
}


oglrap-cmake(){
   local iwd=$PWD

   local bdir=$(oglrap-bdir)
   mkdir -p $bdir
  
   oglrap-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oglrap-idir) \
       $(oglrap-sdir)

   cd $iwd
}

oglrap-make(){
   local iwd=$PWD

   oglrap-bcd 
   make $*

   cd $iwd
}

oglrap-install(){
   oglrap-make install
}



oglrap--()
{
    oglrap-wipe
    oglrap-cmake
    oglrap-make
    oglrap-install

}


oglrap-export()
{
   export SHADER_DIR=$(oglrap-sdir)/gl
} 

oglrap-run(){ 
   local bin=$(oglrap-bindir)/OGLRapTest
   oglrap-export
   $bin $*
}

oglrap-frametest()
{
   local path=${1:-/tmp/teapot.ppm}
   shift
   local bin=$(oglrap-bindir)/FrameTest
   oglrap-export
   $LLDB $bin $path $*
}

oglrap-frametest-lldb()
{
   LLDB=lldb oglrap-frametest $*
}

oglrap-progtest()
{
   SHADER_DIR=~/env/graphics/ggeoview/gl $(oglrap-bindir)/ProgTest
}
