// ./logTest.sh

#include <cstdlib>
#include <array>
#include "NP.hh"

#define KLUDGE_FASTMATH_LOGF(u) (u < 0.998f ? __logf(u) : __logf(u) - 0.46735790f*1e-7f )

const char* FOLD = getenv("FOLD") ? getenv("FOLD") : "/tmp" ; 


__global__ void test_log_(double* dd, unsigned ni, unsigned nj)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    double d = double(ix)/double(ni-1) ; 
    float  f = float(d) ;

    double d0 = -1.*logf( d );
    float  f0 = -1.f*logf( f );
    float  f1 = -1.f*__logf( f );   
    float  f1k = -1.f*KLUDGE_FASTMATH_LOGF(f) ; 

    dd[ix*nj+0] = d ; 
    dd[ix*nj+1] = d0 ; 
    dd[ix*nj+2] = f0 ; 
    dd[ix*nj+3] = f1 ; 
    dd[ix*nj+4] = f1k ; 

    //printf("//test_log  (ix,iy,ni) (%2d, %2d, %2d) \n", ix, iy, ni );
}


void ConfigureLaunch(dim3& numBlocks, dim3& threadsPerBlock, unsigned width )
{ 
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = 1 ; 
    numBlocks.z = 1 ; 
}



void test_log_dev()
{
    unsigned ni = 1000001 ; 
    unsigned nj = 5 ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch(numBlocks, threadsPerBlock, ni ); 


    NP* h = NP::Make<double>( ni, nj ) ; 
    unsigned arr_bytes = h->arr_bytes() ; 
    double* hh = h->values<double>(); 

    double* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    test_log_<<<numBlocks,threadsPerBlock>>>(dd, ni, nj );  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save(FOLD,"dev_scan.npy"); 
}

void test_log_host(double sc)
{
     std::cout 
         << " sc " << std::setw(10) << std::fixed << std::setprecision(8) << sc
         << std::endl 
         ;

     std::array<double, 8> aa = {   1e-8,    1e-7,   1e-6,   1e-5,   1e-4,   1e-3,   1e-2,   1e-1 } ; 
     std::array<double, 8> uu = {1.-1e-8, 1.-1e-7,1.-1e-6,1.-1e-5,1.-1e-4,1.-1e-3,1.-1e-2,1.-1e-1 } ; 
     for(unsigned i=0 ; i < uu.size() ; i++)
     {
         double a  = aa[i] ; 
         double u  = uu[i] ;
         float  fu = uu[i] ; 
         //float  fa = 1.f - fu ; 

         double logu  = -log(u) ; 
         float  logfu = -log(fu) ; 
         float  alogfu  = a  ;     
         //float  alogfu  = fu > 0.999f ? a : logfu  ;     

         //  USING THIS APPROX LOOKS LIKE IT DOES BETTER
         //  BUT IT IS CHEATING AS IT USES DOUBLE PRECISION a 
         //  BUT IN REALITY NEED TO GET a  BY SUBTRACTING FROM 1.f   
         //
         //     -ln(1-x) is very close to x for small x
         // ie  -ln(u) is very close to 1-u for u close to 1.
         //

         double cf = logu - double(logfu); 
         double acf = logu - double(alogfu); 

         std::cout 
             << " a " << std::setw(10) << std::fixed << std::setprecision(8) << a 
             << " u " << std::setw(10) << std::fixed << std::setprecision(8) << u 
             << " fu " << std::setw(10) << std::fixed << std::setprecision(8) << fu 
             << " -log(u)*sc " << std::setw(10) << std::fixed << std::setprecision(8) << logu*sc
             << " -log(fu)*sc " << std::setw(10) << std::fixed << std::setprecision(8) << logfu*sc
             << " alogfu*sc " << std::setw(10) << std::fixed << std::setprecision(8) << alogfu*sc
             << " cf*sc " << std::setw(10) << std::fixed << std::setprecision(8) << cf*sc
             << " acf*sc " << std::setw(10) << std::fixed << std::setprecision(8) << acf*sc
             << std::endl
             ;  
     }
}

int main()
{
    test_log_dev();
    //test_log_host(1.);
    //test_log_host(1e7);
    return 0 ; 
}


