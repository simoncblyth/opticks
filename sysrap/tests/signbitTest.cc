// name=signbitTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <vector>
#include <cmath>
#include <cstdio>
#include <cassert>



void check_nan()
{
    unsigned n = 8 ; 
    float* pairs = new float[n*2]  
    {  
        1.f,  1.f,
       -1.f,  1.f, 
        1.f,  0.f, 
       -1.f,  0.f, 
        0.f,  0.f, 
       -0.f,  0.f, 
        0.f, -0.f,
       -0.f, -0.f 
    }; 

    for(unsigned i=0 ; i < n ; i++ )
    {
        float num = pairs[2*i+0] ; 
        float den = pairs[2*i+1] ; 
        float rat = num/den ; 
        bool  neg = signbit(rat); 

        float nrat = -rat ; 
        float rat0 = rat + 0.f ; 
        float rat2 = rat*2.f ; 
        // all expressions with nan yield nan

        bool gt_0 = rat > 0.f ; 
        bool ge_0 = rat >= 0.f ; 
        bool lt_0 = rat < 0.f ; 
        bool le_0 = rat <= 0.f ; 
        bool eq_0 = rat == 0.f ; 
        bool or_expr_0 = num < 100.f || rat > 0.f ;   // the nan always false does NOT spread to short circuit OR result 
        bool or_expr_1 = rat > 0.f || num < 100.f ;  
        bool or_expr_2 = rat > 0.f || num > 100.f ;  


        printf("//signbitTest  i %d num %10.4f den %10.4f rat %10.4f signbit %d  -rat %10.4f  rat+0.f %10.4f rat*2.f %10.4f ", i, num, den, rat, neg, nrat, rat0, rat2 ); 
        printf(" gt_0 %d ge_0 %d lt_0 %d le_0 %d eq_0 %d or_expr_0 %d or_expr_1 %d or_expr_2 %d  \n", gt_0, ge_0, lt_0, le_0, eq_0, or_expr_0, or_expr_1, or_expr_2 ); 

        if(std::isnan(rat))    
        {
            //printf("//isnan : all comparisons with nan return false \n"); 
            assert( gt_0 == false ); 
            assert( ge_0 == false ); 
            assert( lt_0 == false ); 
            assert( le_0 == false ); 
            assert( eq_0 == false ); 
        }

    } 
}


void check_inf()
{
    float inf = 1.f/0.f ; 
    float ninf = -inf; 

    float inf_times_zero = inf*0.f ; 
    float inf_plus_zero = inf+0.f ; 
    float inf_times_one = inf*1.f ; 
    float inf_plus_one = inf+1.f ; 

    float ninf_times_zero = ninf*0.f ; 
    float ninf_plus_zero = ninf+0.f ; 
    float ninf_times_one = ninf*1.f ; 
    float ninf_plus_one = ninf+1.f ; 

    float ninf_times_minus_one = ninf*-1.f ; 


    printf("// inf      %10.4f \n", inf ); 
    printf("// ninf     %10.4f \n", ninf ); 

    printf("// inf_times_zero %10.4f \n", inf_times_zero ); 
    printf("// inf_plus_zero %10.4f \n",  inf_plus_zero ); 
    printf("// inf_times_one %10.4f \n", inf_times_one ); 
    printf("// inf_plus_one %10.4f \n",  inf_plus_one ); 

    assert( std::isnan(inf_times_zero) ); 
    assert( std::isinf(inf_plus_zero) ); 
    assert( std::isinf(inf_times_one) ); 
    assert( std::isinf(inf_plus_one) ); 

    printf("// ninf_times_zero %10.4f \n", ninf_times_zero ); 
    printf("// ninf_times_minus_one %10.4f \n", ninf_times_minus_one ); 
    printf("// ninf_plus_zero %10.4f \n",  ninf_plus_zero ); 
    printf("// ninf_times_one %10.4f \n", ninf_times_one ); 
    printf("// ninf_plus_one %10.4f \n",  ninf_plus_one ); 

    assert( std::isnan(ninf_times_zero) ); 
    assert( std::isinf(ninf_plus_zero) ); 
    assert( std::isinf(ninf_times_one) ); 
    assert( std::isinf(ninf_plus_one) ); 
}




int main(int argc, char** argv)
{
    // check_nan(); 
    check_inf(); 
    return 0 ; 
}
