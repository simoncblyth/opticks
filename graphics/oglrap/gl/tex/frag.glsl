#version 400 core
#pragma debug(on)

in vec3 colour;
in vec2 texcoord;

out vec4 frag_colour;

uniform sampler2D ColorTex ;
uniform sampler2D DepthTex ;

void main () 
{
   //float depth = texture(DepthTex, texcoord).r ;

   frag_colour = texture(ColorTex, texcoord);

   //float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu

   gl_FragDepth = frag_colour.w  ;

   // frag_colour = vec4( depth, depth, depth, 1.0 );  
   // vizualize fragment depth, the closer you get to geometry the darker it gets 
   // reaching black just before being near clipped

   frag_colour.w = 1.0 ; 

   //gl_FragDepth = 1.1 ;   // black
   //gl_FragDepth = 1.0 ;   // black
   //gl_FragDepth = 0.999 ; //  visible geometry
   //gl_FragDepth = 0.0   ; //  visible geometry 
}

//
// http://www.roxlu.com/2014/036/rendering-the-depth-buffer
//
