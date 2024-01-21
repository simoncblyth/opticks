#pragma once

/**
qmultifilm.h
==============

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QMULTIFILM_METHOD __device__
#else
   #define QMULTIFILM_METHOD 
#endif 

struct qmultifilm
{
    cudaTextureObject_t nnvt_normal_tex[2];
    cudaTextureObject_t nnvt_highqe_tex[2];
    cudaTextureObject_t hama_tex[2];
    
    quad4*              nnvt_normal_meta[2];
    quad4*              nnvt_highqe_meta[2];
    quad4*              hama_meta[2];
 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QMULTIFILM_METHOD float4 lookup(unsigned pmtType, float nm, float aoi);  
#else
    qmultifilm(){}
#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)

/**
qmultifilm::lookup
-----------------------

x --> width  --> nj --> aoi
y --> height --> ni --> wavelength 

**/

inline QMULTIFILM_METHOD float4 qmultifilm::lookup(unsigned pmtType, float nm, float aoi)
{
    const unsigned& nx = hama_meta[0]->q0.u.x  ; // aoi
    const unsigned& ny = hama_meta[0]->q0.u.y  ; // wavelength
 	
	unsigned half_nx = nx/2; 
 
    float wv_low  = hama_meta[0]->q2.f.x;
    float wv_high = hama_meta[0]->q2.f.y;
    float wv_step = (wv_high - wv_low)/(ny-1);   

    float aoi_low = hama_meta[0]->q1.f.x;
    //float aoi_high= hama_meta[0]->q1.f.y;
	
    //float aoi_step= (aoi_high-aoi_low)/(nx-1);   
    
    float aoi_sublow = hama_meta[0]->q1.f.z;
    float aoi_subhigh = hama_meta[0]->q1.f.w;
    float aoi_substep = (aoi_subhigh-aoi_sublow)/(nx-1);
   
    int resolution = ( aoi > aoi_sublow && aoi < aoi_subhigh )? 1 : 0 ;
    int tex_index = resolution ;
   
    cudaTextureObject_t tex = 0 ;
    switch(pmtType)
    {
        case 0: tex = nnvt_normal_tex[tex_index] ; break;
        case 1: tex = hama_tex[tex_index]        ; break;
        case 2: tex = nnvt_highqe_tex[tex_index] ; break;
    }

	float minus_epsilon = -1e-6f;
	float plus_epsilon  = 1e-6f;

	float x = 0.f;
	float y = 0.f;
	
	float aoi_halfstep = (minus_epsilon - aoi_low)/(half_nx - 1);
	if(resolution == 1){
		x = ((aoi - aoi_sublow)/aoi_substep+0.5f)/float(nx);
	}else{

		if(aoi < minus_epsilon){
			x = ((aoi - aoi_low)/aoi_halfstep + 0.5f)/float(nx);
		}
		if(aoi > plus_epsilon){
			x = (half_nx + (aoi-plus_epsilon)/aoi_halfstep + 0.5f)/float(nx);
		}
		if(aoi >= minus_epsilon and aoi <= plus_epsilon){
			x = (half_nx + (aoi - minus_epsilon)/(2*plus_epsilon) + 0.5f)/float(nx);
		}

	}
	
    //float x = resolution == 1 ? ((aoi - aoi_sublow)/aoi_substep +0.5f)/float(nx) : ((aoi - aoi_low)/aoi_step +0.5f)/float(nx) ;
    y = ((nm - wv_low)/wv_step + 0.5f)/float(ny);

    float4 value = tex2D<float4>(tex,x,y);

    return value;
}

#endif

