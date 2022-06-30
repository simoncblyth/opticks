// ./logTest.sh

#include <array>
#include "NP.hh"

__global__ void test_log_(double* dd)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nx = blockDim.x ; 

    double d = double(ix)/double(nx-1) ; 
    float  f = float(d) ;

    double d0 = -1.*log( d );
    float  f0 = -1.f*log( f );

    dd[ix*4+0] = d ; 
    dd[ix*4+1] = d0 ; 
    dd[ix*4+2] = f0 ; 
    dd[ix*4+3] = 0. ; 

    //printf("//test_log  (ix,iy,nx) (%2d, %2d, %2d) \n", ix, iy, nx );
}

void test_log_dev()
{
    unsigned ni = 1001 ; 
    unsigned nj = 4 ; 

    dim3 block(ni,1); 
    dim3 grid(1,1);

    NP* h = NP::Make<double>( ni, nj ) ; 
    unsigned arr_bytes = h->arr_bytes() ; 
    double* hh = h->values<double>(); 

    double* dd = nullptr ; 
    cudaMalloc(reinterpret_cast<void**>( &dd ), arr_bytes );     

    test_log_<<<grid,block>>>(dd);  

    cudaMemcpy( hh, dd, arr_bytes, cudaMemcpyDeviceToHost ) ; 
    cudaDeviceSynchronize();

    h->save("/tmp/logTest.npy"); 
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
    //test_log_dev();
    test_log_host(1.);
    test_log_host(1e7);
    return 0 ; 
}


