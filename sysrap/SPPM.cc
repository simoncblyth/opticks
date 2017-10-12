
#include <iostream>
#include <fstream>

#include "SPPM.hh"


// /Developer/OptiX_380/SDK/primeMultiGpu/primeCommon.cpp

//  https://en.wikipedia.org/wiki/Netpbm_format

/*



PPM uses 24 bits per pixel: 8 for red, 8 for green, 8 for blue.


*/



void SPPM::write( const char* filename, const float* image, int width, int height, int ncomp )
{

  std::ofstream out( filename, std::ios::out | std::ios::binary );
  if( !out ) 
  {
    std::cerr << "Cannot open file " << filename << "'" << std::endl;
    return;
  }

  out << "P6\n" << width << " " << height << "\n255" << std::endl;

  for( int y=height-1; y >= 0; --y ) // flip vertically
  {   
    for( int x = 0; x < width*ncomp; ++x ) 
    {   
      float val = image[y*width*ncomp + x]; 
      unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val*255.0f );
      out.put( cval );
    }   
  }
   
  std::cout << "Wrote file " << filename << std::endl;


}



void SPPM::write( const char* filename, const unsigned char* image, int width, int height, int ncomp )
{

  FILE * fp;
  fp = fopen(filename, "wb");


  fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

  unsigned char* data = new unsigned char[height*width*3] ; 

  for( int y=height-1; y >= 0; --y ) // flip vertically
  {   
  for( int x=0; x < width ; ++x ) 
  {   
     *(data + (y*width+x)*3+0) = image[(y*width+x)*ncomp+0] ;   
     *(data + (y*width+x)*3+1) = image[(y*width+x)*ncomp+1] ;   
     *(data + (y*width+x)*3+2) = image[(y*width+x)*ncomp+2] ;   
  }
  } 


  fwrite(data, sizeof(unsigned char)*height*width*3, 1, fp);
  fclose(fp);  


  std::cout << "Wrote file " << filename << std::endl;

  delete[] data;

}





