#version 400

uniform mat4 ModelViewProjection ;

layout (triangles) in;
layout (line_strip, max_vertices=2) out;

//
//  vertex ordering matching OptiX: intersect_triangle_branchless
//    /Developer/OptiX/include/optixu/optixu_math_namespace.h  
//

void main()
{
   vec3 p0 = gl_in[0].gl_Position.xyz;
   vec3 p1 = gl_in[1].gl_Position.xyz;
   vec3 p2 = gl_in[2].gl_Position.xyz;

   vec3 pc = (p0 + p1 + p2)/3.0 ;  
 
   vec3 e0 = p1 - p0; 
   vec3 e1 = p0 - p2;    
   vec3 no = normalize(cross( e1, e0 )); 

   gl_Position = ModelViewProjection * vec4 (pc, 1.0);
   EmitVertex();

   gl_Position = ModelViewProjection * vec4 (pc + no*1000., 1.0);
   EmitVertex();

   EndPrimitive();
}

