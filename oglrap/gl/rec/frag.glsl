#version 410 core

in vec4 fcolor;
out vec4 frag_color;

//uniform  vec4 ScanParam ;
//uniform ivec4 NrmParam ;
//more efficient to skip in geometry shader rather than in fragment, if possible

void main () 
{
    frag_color = fcolor ;

    //if(NrmParam.z == 1)
    //{
    //    if(gl_FragCoord.z < ScanParam.x || gl_FragCoord.z > ScanParam.y ) discard ;
    //}

}


