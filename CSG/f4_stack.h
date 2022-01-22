#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define F4_FUNC __forceinline__ __device__ __host__
#else
#    define F4_FUNC inline
#endif


/**

**/


struct F4_Stack 
{
    float4 data  ; 
    int curr ;

    int push(float f)
    {
        curr += 1 ;    // curr must start at -1  
        switch(curr)
        { 
            case 0:  data.x = f ; 
            case 1:  data.y = f ; 
            case 2:  data.z = f ; 
            case 3:  data.w = f ; 
        }
        return 0 ; 
    } 
    int pop(float& f0 )
    {
        switch(curr)
        {
            case 0: f0 = data.x ; break ; 
            case 1: f0 = data.y ; break ; 
            case 2: f0 = data.z ; break ; 
            case 3: f0 = data.w ; break ; 
        }
        curr -= 1  ; 
        return 0 ; 
    }

    int push2(float f0, float f1)
    {
        switch(curr)
        {
            case -1: data.x = f0 ; data.y = f1 ; break ; 
            case  0: data.y = f0 ; data.z = f1 ; break ; 
            case  1: data.z = f0 ; data.w = f1 ; break ; 
        }
        curr += 2 ;    // curr must start at -1  
        return 0 ; 
    }

    int pop2(float& f0, float& f1)
    {
        switch(curr)
        {
           case 0: f0 = -1.f   ; f1 = -1.f   ; break ; 
           case 1: f0 = data.x ; f1 = data.y ; break ; 
           case 2: f0 = data.y ; f1 = data.z ; break ; 
           case 3: f0 = data.z ; f1 = data.w ; break ; 
        }
        curr -= 2  ; 
        return 0 ; 
    }
};



