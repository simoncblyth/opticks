#version 410 core
uniform mat4 MVP;
in vec3 vNrm;
in vec3 vPos;
out vec4 v_color;

void main()
{
    gl_Position = MVP * vec4(vPos, 1.0);
    v_color = vec4( vNrm*0.5 + 0.5, 1.0 );
}



