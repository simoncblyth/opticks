#version 400 core
//#pragma debug(on)

in vec3 colour;
in vec2 texcoord;

out vec4 frag_colour;

uniform  vec4 ScanParam ;
uniform vec4 ClipPlane ;
uniform ivec4 NrmParam ;

uniform sampler2D ColorTex ;

void main () 
{
   frag_colour = texture(ColorTex, texcoord);
   float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu
   frag_colour.w = 1.0 ; 

   gl_FragDepth = depth  ;

   if(NrmParam.z == 1)
   {
        if(depth < ScanParam.x || depth > ScanParam.y ) discard ;
   } 
}

//
// http://www.roxlu.com/2014/036/rendering-the-depth-buffer
//
// gl_FragDepth = 1.1 ;   // black
// gl_FragDepth = 1.0 ;   // black
// gl_FragDepth = 0.999 ; //  visible geometry
// gl_FragDepth = 0.0   ; //  visible geometry 
//
// frag_colour = vec4( depth, depth, depth, 1.0 );  
// vizualize fragment depth, the closer you get to geometry the darker it gets 
// reaching black just before being near clipped
//
