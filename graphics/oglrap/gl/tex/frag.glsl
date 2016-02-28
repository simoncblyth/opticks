#version 400 core
#pragma debug(on)

in vec3 colour;
in vec2 texcoord;

out vec4 frag_colour;

uniform sampler2D ColorTex ;
uniform sampler2D DepthTex ;

void main () 
{
   float depth = texture(DepthTex, texcoord).r ;

   //frag_colour = vec4( depth, depth, depth, 1.0 ); gives black 

   frag_colour = texture(ColorTex, texcoord);

   //frag_colour.r = 0.0 ; 
   //frag_colour.r = depth ; 
   //  with depth and with 0. look the same 

   gl_FragDepth = depth ;

   //gl_FragDepth = 1.1 ; // black
   //gl_FragDepth = 1.0 ; // black
   //gl_FragDepth = 0.999 ; //  visible geometry
   //gl_FragDepth = 0.0   ; //  visible geometry 

   //frag_colour = vec4 (colour, 1.0);
   //frag_colour = vec4 (1.0,0.0,0.0, 1.0);

}

//
// http://www.roxlu.com/2014/036/rendering-the-depth-buffer
//
