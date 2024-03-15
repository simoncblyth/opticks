#version 410 core
uniform mat4 MVP;

in vec3 vNrm;
in vec3 vPos;
in mat4 vInstanceTransform ; 

out vec4 v_color;

void main()
{
    gl_Position = MVP * vInstanceTransform * vec4(vPos, 1.0);

    //v_color = vec4( vNrm*0.5 + 0.5, 1.0 );
    v_color = vec4( clamp( vNrm*0.5 + 0.8, 0.0, 1.0) , 1.0 );

}



