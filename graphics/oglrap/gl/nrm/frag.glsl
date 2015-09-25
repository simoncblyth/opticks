#version 400

in vec4 colour;
out vec4 frag_colour;

uniform ivec4 NrmParam ;

uniform vec4 ColorDomain ;
uniform sampler1D Colors ;



void main () 
{
    
    if(NrmParam.y == 3)
    {
        frag_colour = texture(Colors, float(64 + gl_PrimitiveID % int(ColorDomain.z))/ColorDomain.y ) ;
    }
    else
    {
        frag_colour = colour ;
    }

}


