/**

~/o/sysrap/tests/scuda_test.sh 

**/


#include "ssys.h"
#include "scuda.h"

struct scuda_test
{
    static const char* TEST ; 
    static int cross(); 
    static int indexed(); 
    static int efloat(); 
    static int serial(); 
    static int uint4_increment(); 
    static int uint4_skipahead(); 

    static int main(); 
}; 

const char* scuda_test::TEST = ssys::getenvvar("TEST","ALL"); 



int scuda_test::cross()
{
    float3 mom = make_float3( -0.2166f,-0.9745f, 0.0578f ); 
    float3 oriented_normal = make_float3(  0.2166f,0.9745f, -0.0578f ); 
    float3 trans = ::cross( mom, oriented_normal );    
    printf("// trans (%10.7f, %10.7f, %10.7f) \n", trans.x, trans.y, trans.z ); 

    float trans_mag2 = dot(trans, trans) ; 
    bool trans_mag2_is_zero = trans_mag2  == 0.f ; 
    printf("// trans_mag2 %10.9f trans_mag2_is_zero %d \n", trans_mag2, trans_mag2_is_zero );  

    float3 A_trans = normalize(trans) ; 
    printf("// A_trans (%10.7f, %10.7f, %10.7f) \n", A_trans.x, A_trans.y, A_trans.z ); 
    return 0 ; 
}


int scuda_test::indexed()
{
    float4 v = make_float4( 0.f , 1.f, 2.f, 3.f ); 
    const float* vv = (const float*)&v ; 
    printf("//test_indexed vv[0] %10.4f vv[1] %10.4f vv[2] %10.4f vv[3] %10.4f \n",vv[0], vv[1], vv[2], vv[3] ); 
    return 0 ; 
}


int scuda_test::efloat()
{
    float f = scuda::efloat("f",101.102f); 
    printf("//test_efloat f (%10.4f) \n", f ); 

    float3 v3 = scuda::efloat3("v3","3,33,333"); 
    printf("//test_efloat3 v3 (%10.4f %10.4f %10.4f) \n", v3.x, v3.y, v3.z ); 

    float4 v4 = scuda::efloat4("v4","4,44,444,4444"); 
    printf("//test_efloat4 v4 (%10.4f %10.4f %10.4f %10.4f) \n", v4.x, v4.y, v4.z, v4.w ); 

    float3 v3n = scuda::efloat3n("v3n","1,1,1"); 
    printf("//test_efloat3n v3n (%10.4f %10.4f %10.4f) \n", v3n.x, v3n.y, v3n.z ); 
    return 0 ; 
}


int scuda_test::serial()
{
    float4 v = make_float4( 0.001f , 1.004f, 2.00008f, 3.0006f ); 
    std::cout << " v " << v << std::endl ; 

    std::cout << " scuda::serialize(v) " <<  scuda::serialize(v) << std::endl ; 
    return 0 ; 
}





void increment(uint4& ctr )
{
   if(++ctr.x==0) 
   if(++ctr.y==0) 
   if(++ctr.z==0) 
   ++ctr.w ; 

}

void increment_n(uint4& ctr, unsigned long long n, bool dump)
{
   for(unsigned long long i=0ull ; i < n ; i++) increment(ctr); 
   if(dump) std::cout << "increment_n " << std::setw(16) << n  << " : " << ctr << "\n" ; 
}




void skipahead(uint4& ctr, unsigned long long n)
{
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n>>32);

    ctr.x += nlo;
    if( ctr.x < nlo ) nhi++;

    ctr.y += nhi;
    if(nhi <= ctr.y) return;
    if(ctr.z) return;
    ++ctr.w;

}
void _skipahead(uint4& ctr, unsigned long long n)
{
    skipahead(ctr, n); 
    std::cout << "_skipahead " << std::hex << std::setw(16) << n << " : " << ctr << "\n" ; 
}


int scuda_test::uint4_increment()
{
    std::cout << "[scuda_test::uint4_increment\n" ; 
    uint4 ctr = {};
    increment_n(ctr, 1000, true ); 
    std::cout << "]scuda_test::uint4_increment\n" ; 
    return 0 ; 
}

int scuda_test::uint4_skipahead()
{
    typedef unsigned long long ULL ; 
    std::cout << "[scuda_test::uint4_skipahead \n" ; 
    std::array<ULL,3> skip = { 1000ull, 0xffffffffull, 0xffffffffffffffffull };
    for(int i=0 ; i < int(skip.size()) ; i++ )
    {
        ULL n = skip[i];  

        uint4 ctr0 = {};
        uint4 ctr1 = {};

        _skipahead(ctr0, n ); 
        _skipahead(ctr0, n ); 
        _skipahead(ctr0, n ); 

        increment_n(ctr1, n, true ); 
        increment_n(ctr1, n, true ); 
        increment_n(ctr1, n, true ); 
         
    }
    std::cout << "]scuda_test::uint4_skipahead \n" ; 
    return 0 ; 
}


int scuda_test::main()
{
    bool ALL = strcmp(TEST, "ALL")==0 ; 
    int rc = 0 ; 
    if(ALL||0==strcmp(TEST,"cross")) rc += cross();
    if(ALL||0==strcmp(TEST,"indexed")) rc += indexed();
    if(ALL||0==strcmp(TEST,"efloat")) rc += efloat();
    if(ALL||0==strcmp(TEST,"serial")) rc += serial();
    if(ALL||0!=strstr(TEST,"uint4_increment")) rc += uint4_increment();
    if(ALL||0!=strstr(TEST,"uint4_skipahead")) rc += uint4_skipahead();

    return rc ; 
}

int main(){  return scuda_test::main() ; }


