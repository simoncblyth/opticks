// name=variable_size_object_may_not_be_initialized ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 
// name=variable_size_object_may_not_be_initialized ; gcc $name.cc -std=c++11 -lstdc++ -DFIX -o /tmp/$name && /tmp/$name 

#include <iostream>

struct RecTimeLikeAlg
{
    int m_Algorithm; 
#ifdef FIX
    static constexpr int nbins_z = 10 ; 
#else
    int nbins_z ; 
#endif

    bool Load_LikeFun() ;
};

bool RecTimeLikeAlg::Load_LikeFun()
{
    if( m_Algorithm == 2 )
    {
#ifdef FIX
        //nbins_z = 10;
#else
        nbins_z = 10;
#endif
        double zbinning[nbins_z+1] = {0, 8.21561, 10.351, 11.849, 13.0415, 14.0485, 14.9288, 15.7159, 16.4312, 17.0892, 17.7};
        for (int j=0; j<=nbins_z; j++){
            std::cout << zbinning[j] << std::endl ;
        }    
    }    
    return true ; 
}

int main()
{
   RecTimeLikeAlg rtla ; 
   rtla.m_Algorithm = 2 ;
   rtla.Load_LikeFun();  
   return 0 ; 
}

/**

variable_size_object_may_not_be_initialized.cc:19:25: error: variable-sized object may not be initialized
        double zbinning[nbins_z+1] = {0, 8.21561, 10.351, 11.849, 13.0415, 14.0485, 14.9288, 15.7159, 16.4312, 17.0892, 17.7};
                        ^~~~~~~~~
1 error generated.



**/

