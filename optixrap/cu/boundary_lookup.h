/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

/**
boundary_lookup.h
-------------------

Convenience functions to reading properties from the boundary texture.

The boundary texture is constructed from material and surface 
properties collected into ggeo/GBndLib and then converted into 
a GPU texture by optixrap/OBndLib::convert

* see cu/boundaryLookupTest.cu for usage and testing


**/

// just some BOUNDARY_ defines specifying boundary texture shape
#include "GPropertyLib.hh"

rtTextureSampler<float4, 2>  boundary_texture ;
rtDeclareVariable(float4, boundary_domain, , );
rtDeclareVariable(float4, boundary_domain_reciprocal, , );
rtDeclareVariable(uint4,  boundary_bounds, , );
rtDeclareVariable(uint4,  boundary_texture_dim, , );


static __device__ __inline__ float boundary_sample_reciprocal_domain(const float& u)
{
    // return wavelength, from uniform sampling of 1/wavelength[::-1] domain
    // need to flip to match Geant4 energy sampling, see boundary_lookup.py 
    //float iw = lerp( boundary_domain_reciprocal.x , boundary_domain_reciprocal.y, u ) ;
    float iw = lerp( boundary_domain_reciprocal.y , boundary_domain_reciprocal.x, u ) ;
    return 1.f/iw ;  
}

static __device__ __inline__ float boundary_sample_reciprocal_domain_v3(const float& u)
{
    // see boundary_lookup.py  
    //
    // using this for Cerenkov results in a bug, should be using the energy/wavelength range 
    // from RINDEX properties
   
    float a = boundary_domain.x ; 
    float b = boundary_domain.y ; 
    return a*b/lerp( a, b, u ) ;
}


static __device__ __inline__ float boundary_sample_domain(const float& u)
{
    // return wavelength, from uniform sampling of wavelength domain
    return lerp( boundary_domain.x , boundary_domain.y, u ) ;
}

static __device__ __inline__ unsigned boundary_lookup_ijk( unsigned int i, unsigned int j, unsigned int k)
{
    /*
        i   boundary index 0..ni-1  
        j   qwn index      0..nj-1  0,1,2,3
        k   grp index      0..nk-1  0,1

        (ni-1)*nj*nk + (nj-1)*nk + (nk-1) =  ni*nj*nk - nj*nk + nj*nk - nk + nk - 1 ==> ni*nj*nk -1   
    */
    //unsigned ni = boundary_texture_dim.z ;    
    unsigned nj = BOUNDARY_NUM_MATSUR ;     
    unsigned nk = BOUNDARY_NUM_FLOAT4 ;     
    unsigned iy = i*nj*nk + j*nk + k ;      // iy: 0->ny-1
    return iy ; 
}

/**

Its advantageous to keep k (grp index) separate as the 
number of groups is the thing that can be easily changed. 

**/

static __device__ __inline__ unsigned boundary_lookup_linek( unsigned int line, unsigned int k)
{
    // line :    i*nj + j    
    unsigned nk = BOUNDARY_NUM_FLOAT4 ;       // 2 
    unsigned iy = line*nk + k ;      // iy: 0->ny-1
    return iy ; 
}

static __device__ __inline__ float4 boundary_lookup( float nm, unsigned int i, unsigned int j, unsigned int k)
{
    // nm:  wavelength in nanometers

    unsigned nx = boundary_texture_dim.x ;  //  (ni*nj*nk)  ix: 0->nx-1
    unsigned ny = boundary_texture_dim.y ;
    unsigned ix = unsigned((nm - boundary_domain.x)/boundary_domain.z) ;    // (nm-80.)/20. -> 0..38 
    unsigned iy = boundary_lookup_ijk(i, j, k );

    float x = (float(ix)+0.5f)/float(nx) ; 
    float y = (float(iy)+0.5f)/float(ny) ; 

    float4 val = tex2D(boundary_texture, x, y );
    //rtPrintf("boundary_lookup (%d,%d) (%10.4f,%10.4f) -> (%10.4f,%10.4f,%10.4f,%10.4f)  \n", ix, iy, x, y, val.x, val.y, val.z, val.w);
    return val ;
}

static __device__ __inline__ float4 boundary_lookup( float nm, unsigned int line, unsigned int k)
{
    // nm:  wavelength in nanometers

    unsigned nx = boundary_texture_dim.x ;  //  (ni*nj*nk)  ix: 0->nx-1
    unsigned ny = boundary_texture_dim.y ;

    //unsigned ix = unsigned((nm - boundary_domain.x)/boundary_domain.z) ;    // (nm-80.)/20. -> unsigned(0..38) 
    //float x = (float(ix)+0.5f)/float(nx) ; 

    float fx = (nm - boundary_domain.x)/boundary_domain.z ;    // (nm-80.)/20. -> float(0..38) 
    float x = (fx+0.5f)/float(nx) ; 

    unsigned iy = boundary_lookup_linek(line, k );
    float y = (float(iy)+0.5f)/float(ny) ; 

    float4 val = tex2D(boundary_texture, x, y );
    //rtPrintf("boundary_lookup (%d,%d) (%10.4f,%10.4f) -> (%10.4f,%10.4f,%10.4f,%10.4f)  \n", ix, iy, x, y, val.x, val.y, val.z, val.w);
    return val ;
}


