#pragma once
/**
qbnd.h
=========

bnd and optical are closely related so they must be kept together

**/

enum { _BOUNDARY_NUM_MATSUR = 4,  _BOUNDARY_NUM_FLOAT4 = 2 }; 

enum {
    OMAT,
    OSUR,
    ISUR,
    IMAT 
};


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QBND_METHOD __device__
#else
   #define QBND_METHOD 
#endif 

struct quad4 ; 
struct qstate ; 


struct qbnd
{
    cudaTextureObject_t boundary_tex ; 
    quad4*              boundary_meta ; 
    unsigned            boundary_tex_MaterialLine_Water ;
    unsigned            boundary_tex_MaterialLine_LS ; 
    quad*               optical ;  



#if defined(__CUDACC__) || defined(__CUDABE__)
    QBND_METHOD float4  boundary_lookup( unsigned ix, unsigned iy ); 
    QBND_METHOD float4  boundary_lookup( float nm, unsigned line, unsigned k ); 
    QBND_METHOD void    fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta, unsigned idx ); 
#endif

}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
/**
qbnd::boundary_lookup ix iy : Low level integer addressing lookup
--------------------------------------------------------------------

**/

inline QBND_METHOD float4 qbnd::boundary_lookup( unsigned ix, unsigned iy )
{
    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;
    float4 props = tex2D<float4>( boundary_tex, x, y );     
    return props ; 
}

/**
qbnd::boundary_lookup nm line k 
----------------------------------

nm:    float wavelength 
line:  4*boundary_index + OMAT/OSUR/ISUR/IMAT   (0/1/2/3)
k   :  property group index 0/1 

return float4 props 

boundary_meta is required to configure access to the texture, 
it is uploaded by QTex::uploadMeta but requires calls to 


QTex::init automatically sets these from tex dimensions

   q0.u.x : nx width  (eg wavelength dimension)
   q0.u.y : ny hright (eg line dimension)

QTex::setMetaDomainX::

   q1.f.x : nm0  wavelength minimum in nm 
   q1.f.y : -
   q1.f.z : nms  wavelength step size in nm   

QTex::setMetaDomainY::

   q2.f.x : 
   q2.f.y :
   q2.f.z :


**/
inline QBND_METHOD float4 qbnd::boundary_lookup( float nm, unsigned line, unsigned k )
{
    //printf("//qbnd.boundary_lookup nm %10.4f line %d k %d boundary_meta %p  \n", nm, line, k, boundary_meta  ); 

    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    const float& nm0 = boundary_meta->q1.f.x ; 
    const float& nms = boundary_meta->q1.f.z ; 

    float fx = (nm - nm0)/nms ;  
    float x = (fx+0.5f)/float(nx) ;   // ?? +0.5f ??

    unsigned iy = _BOUNDARY_NUM_FLOAT4*line + k ;   
    float y = (float(iy)+0.5f)/float(ny) ; 


    float4 props = tex2D<float4>( boundary_tex, x, y );     

    // printf("//qbnd.boundary_lookup nm %10.4f nm0 %10.4f nms %10.4f  x %10.4f nx %d ny %d y %10.4f props.x %10.4f %10.4f %10.4f %10.4f  \n",
    //     nm, nm0, nms, x, nx, ny, y, props.x, props.y, props.z, props.w ); 

    return props ; 
}


/**
qbnd::fill_state
-------------------

Formerly signed the 1-based boundary, now just keeping separate cosTheta to 
orient the use of the boundary so are using 0-based boundary. 

cosTheta < 0.f 
   photon direction is against the surface normal, ie are entering the shape
   
   * formerly this corresponded to -ve boundary 
   * line+OSUR is relevant surface
   * line+OMAT is relevant first material

cosTheta > 0.f 
   photon direction is with the surface normal, ie are exiting the shape
   
   * formerly this corresponded to +ve boundary
   * line+ISUR is relevant surface
   * line+IMAT is relevant first material


NB the line is above the details of the payload (ie how many float4 per matsur) it is just::
 
    boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 


The optical buffer is 4 times the length of the bnd, which allows
convenient access to the material and surface indices starting 
from a texture line.  See notes in:: 

    GBndLib::createOpticalBuffer 
    GBndLib::getOpticalBuf

Notice that s.optical.x and s.index.z are the same thing. 
So half of s.index is extraneous and the m1 index and m2 index 
is not much used.  

Also only one elemnt of m1group2 is actually used 


s.optical.x 
    used to distinguish between : boundary, surface (and in future multifilm)
    
    * currently contains 1-based surface index with 0 meaning "boundary" and anything else "surface"

    * TODO: encode boundary type enum into the high bits of s.optical.x for three way split 
      (perhaps use trigger strings like MULTIFILM in the boundary spec to configure)
      THIS WILL NEED TO BE DONE AT x4 translation level (and repeated in QBndOptical for 
      dynamic boundary adding) 

**/

inline QBND_METHOD void qbnd::fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta, unsigned idx  )
{
    const int line = boundary*_BOUNDARY_NUM_MATSUR ;      // now that are not signing boundary use 0-based

    const int m1_line = cosTheta > 0.f ? line + IMAT : line + OMAT ;   
    const int m2_line = cosTheta > 0.f ? line + OMAT : line + IMAT ;   
    const int su_line = cosTheta > 0.f ? line + ISUR : line + OSUR ;   


    s.material1 = boundary_lookup( wavelength, m1_line, 0);   // refractive_index, absorption_length, scattering_length, reemission_prob
    s.m1group2  = boundary_lookup( wavelength, m1_line, 1);   // group_velocity ,  (unused          , unused           , unused)  
    s.material2 = boundary_lookup( wavelength, m2_line, 0);   // refractive_index, (absorption_length, scattering_length, reemission_prob) only m2:refractive index actually used  
    s.surface   = boundary_lookup( wavelength, su_line, 0);   //  detect,        , absorb            , (reflect_specular), reflect_diffuse     [they add to 1. so one not used] 

    //printf("//qsim.fill_state boundary %d line %d wavelength %10.4f m1_line %d \n", boundary, line, wavelength, m1_line ); 

    s.optical = optical[su_line].u ;   // 1-based-surface-index-0-meaning-boundary/type/finish/value  (type,finish,value not used currently)

    //printf("//qsim.fill_state idx %d boundary %d line %d wavelength %10.4f m1_line %d m2_line %d su_line %d s.optical.x %d \n", 
    //    idx, boundary, line, wavelength, m1_line, m2_line, su_line, s.optical.x ); 

    s.index.x = optical[m1_line].u.x ; // m1 index
    s.index.y = optical[m2_line].u.x ; // m2 index 
    s.index.z = optical[su_line].u.x ; // su index
    s.index.w = 0u ;                   // avoid undefined memory comparison issues

    //printf("//qsim.fill_state \n"); 
}




#endif
