#include "GPropertyLib.hh"

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtTextureSampler<float4, 2>  boundary_texture ;
rtDeclareVariable(float4, boundary_domain, , );
rtDeclareVariable(float4, boundary_domain_reciprocal, , );
rtDeclareVariable(uint4, boundary_bounds, , );

rtDeclareVariable(uint4, boundary_test_args, , );



static __device__ __inline__ float4 wavelength_lookup(float nm, unsigned int line, unsigned int offset )
{
    //   line: identifies material or surface omat/osur/isur/imat of a particular boundary 
    //         range 0 -> (#bnd*4) - 1
    //
    //   offset: is either 0 or 1 picking the property group (float4) 

    // x:low y:high z:step w:mid   tex coords are offset by 0.5 
    // float nmi = (nm - boundary_domain.x)/boundary_domain.z + 0.5f ;   

    // may need 0.5 offsets ?

    float nm_x = (nm - boundary_domain.x + 0.5f)/boundary_domain.z ;   

    float ynum = (BOUNDARY_NUM_FLOAT4*line + offset + 0.5f) ;
    float yden = boundary_bounds.w ; 
    float pr_y = ynum / yden ; 

    if( ynum > yden )
    {
        rtPrintf("wavelength_lookup OUT OF BOUNDS nm %10.4f line %4d offset %4d nm_x %10 ynum/yden/y ( %10.4f %10.4f %10.4f )\n",
            nm,
            line,
            offset,
            nm_x,
            ynum,
            yden,
            pr_y);

        rtPrintf("boundary_bounds (%4u,%4u,%4u,%4u) \n", boundary_bounds.x, boundary_bounds.y, boundary_bounds.z, boundary_bounds.w);
        rtPrintf("boundary_domain (%10.4f,%10.4f,%10.4f,%10.4f) \n", boundary_domain.x, boundary_domain.y, boundary_domain.z, boundary_domain.w);
 
    }

    return tex2D(boundary_texture, nm_x, pr_y ) ;

}


/*

   b = np.load("/tmp/blyth/opticks/GBndLib/GBndLib.npy")

   b.shape

         #float4
            |     ___ wavelength samples
            |    /
   (123, 4, 2, 39, 4)
    |    |          \___ float4 props        
  #bnd   | 
         |
    omat/osur/isur/imat  

*/


static __device__ __inline__ void boundary_dump()
{
   int ni = 123 ; 
   int nj = 4 ; 
   int nk = 2 ; 

   int nx = 39 ; 
   int ny = ni*nj*nk ; 

   for(int i=0 ; i < ni ; i++)
   {
   for(int j=0 ; j < nj ; j++)
   {
   for(int k=0 ; k < nk ; k++)
   {
       unsigned int ix = 0u ; // 0.. nx-1
       unsigned int iy = i*nj*nk + j*nk + k ;  // 0.. ny-1   ie 0..ni*nj*nk-1  

       float x = (float(ix)+0.5f)/float(nx) ; 
       float y = (float(iy)+0.5f)/float(ny) ; 

       float4 pr = tex2D(boundary_texture, x, y );
       rtPrintf(" i:%d j:%d k:%d x:%10.3f y:%10.3f pr  %13.4f %13.4f %13.4f %13.4f  \n",i,j,k,x,y,pr.x,pr.y,pr.z,pr.w);
   }
   }
   }
}


static __device__ __inline__ void boundary_check(unsigned int ibnd, unsigned int jqwn )
{

  for(int i=0 ; i < 39 ; i++ )
  {
     float nm = boundary_domain.x + boundary_domain.z*i ; 

     unsigned int line = ibnd*BOUNDARY_NUM_MATSUR + jqwn ; // #MATSUR=4   jqwn (0/1/2/3 : oma/osur/isur/imat)
     // line : specifies a material or surface within a specific boundary 


    unsigned int offset = 0 ; 

    float nm_x = (nm - boundary_domain.x + 0.5f)/boundary_domain.z ;   

    float ynum0 = (BOUNDARY_NUM_FLOAT4*line + 0 ) ;
    float ynum1 = (BOUNDARY_NUM_FLOAT4*line + 1 ) ;
    float yden = boundary_bounds.w ; 

    float pr_y0 = ynum0 / yden ; 
    float pr_y1 = ynum1 / yden ; 

    float4 pr0 = tex2D(boundary_texture, nm_x, pr_y0 );
    float4 pr1 = tex2D(boundary_texture, nm_x, pr_y1 );


     rtPrintf("wavelength_check nm:%10.3f ibnd %2u jqwn %u line %3u  pr0  %13.4f %13.4f %13.4f %13.4f pr1  %13.4f %13.4f %13.4f %13.4f \n",
          nm, 
          ibnd,
          jqwn, 
          line,
          pr0.x, 
          pr0.y, 
          pr0.z, 
          pr0.w,
          pr1.x, 
          pr1.y, 
          pr1.z, 
          pr1.w
     ); 
  }
}




RT_PROGRAM void boundaryTest()
{
    rtPrintf("boundaryTest.boundary_domain            %10.3f %10.3f %10.3f %10.3f \n ",
          boundary_domain.x, boundary_domain.y, boundary_domain.z, boundary_domain.w );
    rtPrintf("boundaryTest.boundary_domain_reciprocal %10.3f %10.3f %10.3f %10.3f \n ", 
          boundary_domain_reciprocal.x, boundary_domain_reciprocal.y, boundary_domain_reciprocal.z, boundary_domain_reciprocal.w );
    rtPrintf("boundaryTest.boundary_bounds %u %u %u %u \n ", 
          boundary_bounds.x, boundary_bounds.y, boundary_bounds.z, boundary_bounds.w );

    rtPrintf("boundaryTest.boundary_test_args %u %u %u %u \n ", 
          boundary_test_args.x, boundary_test_args.y, boundary_test_args.z, boundary_test_args.w );

   //unsigned int ibnd=boundary_test_args.x ; 
   //unsigned int jqwn=boundary_test_args.y ;  //OMAT:0
   //boundary_check(ibnd, jqwn);
   boundary_dump();
}

RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



