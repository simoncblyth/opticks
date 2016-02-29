#version 400

#incl dynamic.h

in vec4 colour;
out vec4 frag_colour;

uniform  vec4 ScanParam ;
uniform ivec4 NrmParam ;

uniform vec4 ColorDomain ;
uniform sampler1D Colors ;


void main () 
{
    if(NrmParam.y == 3)
    {
        frag_colour = texture(Colors, float(ColorDomain.w + gl_PrimitiveID % int(ColorDomain.z))/ColorDomain.y ) ;
    }
    else if(NrmParam.z == 1)
    {
        frag_colour = colour ;
        if(gl_FragCoord.z < ScanParam.x || gl_FragCoord.z > ScanParam.y ) discard ;
    }
    else
    {
        frag_colour = colour ;
    }

}


