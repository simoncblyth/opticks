#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout( location = 4) in mat4 InstanceTransform ;

out mat4 ITransform ;    
flat out int objectVisible;

void main() 
{    
    vec4 InstancePosition = InstanceTransform[3] ; 
    vec4 IClip = ModelViewProjection * InstancePosition ;    

    //float f = 0.8f ; 
    float f = 1.0f ; 
    objectVisible = 
         ( IClip.x < IClip.w*f && IClip.x > -IClip.w*f  ) &&
         ( IClip.y < IClip.w*f && IClip.y > -IClip.w*f  ) &&
         ( IClip.z < IClip.w*f && IClip.z > -IClip.w*f  ) ? 1 : 0 ; 


    ITransform = InstanceTransform ; 
}   







