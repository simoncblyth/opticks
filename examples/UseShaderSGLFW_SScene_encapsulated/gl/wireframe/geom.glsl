#version 410 core
// https://stackoverflow.com/questions/137629/how-do-you-render-primitives-as-wireframes-in-opengl

layout (triangles) in;
layout (line_strip, max_vertices=3) out;

in vec4 v_color[];  // vertex colors from Vertex Shader
out vec4 g_color ; 

void main(void)
{
    int i;
    for (i = 0; i < gl_in.length(); i++)
    {
        g_color=v_color[i]; 
        gl_Position = gl_in[i].gl_Position; 
        EmitVertex();
    }
    EndPrimitive();
}

