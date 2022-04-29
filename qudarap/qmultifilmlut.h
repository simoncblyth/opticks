#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPROP_METHOD __device__
#else
   #define QPROP_METHOD 
#endif 

/**
qmultifilmlut
=======

**/

#include "sview.h"


struct qmultifilmlut
{

    cudaTextureObject_t nnvt_normal_tex[4];
    cudaTextureObject_t nnvt_highqe_tex[4];
    cudaTextureObject_t hama_tex[4];
    
    quad4*              nnvt_normal_meta[4];
    quad4*              nnvt_highqe_meta[4];
    quad4*              hama_meta[4];
 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QPROP_METHOD float4 multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi);  
#else
    qmultifilmlut()
    {
    }
#endif

}; 





#if defined(__CUDACC__) || defined(__CUDABE__)

/**
qmultifilmlut::multifilm_lookup
-----------------------

x --> width  --> nj --> aoi
y --> height --> ni --> wavelength 

**/

inline QSIM_METHOD float4 qmultifilmlut::multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi){

    const unsigned& nx = hama_meta[0]->q0.u.x  ; // aoi
    const unsigned& ny = hama_meta[0]->q0.u.y  ; // wavelength
   
    float wv_low  = hama_meta[0]->q2.f.x;
    float wv_high = hama_meta[0]->q2.f.y;
    float wv_step = (wv_high - wv_low)/(ny-1);   

    float aoi_low = hama_meta[0]->q1.f.x;
    float aoi_high= hama_meta[0]->q1.f.y;
    float aoi_step= (aoi_high-aoi_low)/(nx-1);   
    
    float aoi_sublow = hama_meta[0]->q1.f.z;
    float aoi_subhigh = hama_meta[0]->q1.f.w;
    float aoi_substep = (aoi_subhigh-aoi_sublow)/(nx-1);
   
    int resolution = ( aoi > aoi_sublow and aoi < aoi_subhigh )? 1 : 0 ;
    int tex_index = boundary * 2 + resolution ;
   
    cudaTextureObject_t tex = 0 ;
    switch(pmtType){
       case 0: tex = nnvt_normal_tex[tex_index] ; break;
       case 1: tex = hama_tex[tex_index]        ; break;
       case 2: tex = nnvt_highqe_tex[tex_index] ; break;
    }
   
    float x = -1.;
    if( resolution == 1 ){   
          x = ((aoi - aoi_sublow)/aoi_substep +0.5f)/float(nx);
    }
    else{
          x = ((aoi - aoi_low)/aoi_step +0.5f)/float(nx);
    }

    float y = ((nm - wv_low)/wv_step + 0.5f)/float(ny);

    float4 value = tex2D<float4>(tex,x,y);
    return value;
}

#endif

