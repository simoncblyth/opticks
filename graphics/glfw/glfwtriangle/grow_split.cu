


struct grow
{
    unsigned int count ; 
    grow(unsigned int count) : count(count) {}

    __host__ __device__ float3 operator()(unsigned int i)
    {
         float s = 0.01f * (count % 100)  ; 
         float3 vec ; 
         switch(i)
         {
            case 0: vec = make_float3( 0.0f,     s,  0.0f) ; break ;
            case 1: vec = make_float3(    s,    -s,  0.0f) ; break ;
            case 2: vec = make_float3(   -s,    -s,  0.0f) ; break ;
         }  

         return vec ; 
    }
};

