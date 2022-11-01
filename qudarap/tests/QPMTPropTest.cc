#include <cuda_runtime.h>
#include "QProp.hh"
#include "JPMTProp.h"
#include "OPTICKS_LOG.hh"

template<typename T>
struct QPMTProp
{
    const QProp<T>* rindex ; 
    const NP* thickness ;  

    QPMTProp( const NP* rindex, const NP* thickness );     

    std::string desc() const ; 
};

template<typename T>
QPMTProp<T>::QPMTProp( const NP* rindex_ , const NP* thickness_ )
    :
    rindex(QProp<T>::Make3D(rindex_)),
    thickness(thickness_)
{
}

template<typename T>
std::string QPMTProp<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "QPMTProp::desc"
       << std::endl
       << "rindex"
       << std::endl
       << rindex->desc()
       << std::endl
       << "thickness"
       << std::endl
       << thickness->desc()
       << std::endl
       ;
    std::string s = ss.str(); 
    return s ;
} 

const char* FOLD = "/tmp/QPMTPropTest" ;

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    JPMTProp pp ; 
    //std::cout << pp.desc() << std::endl ;

    QPMTProp<double> qpp(pp.rindex, pp.thickness) ;   
    std::cout << qpp.desc() << std::endl ; 

    qpp.rindex->lookup_scan( 1.55, 15.5, 100u, FOLD, "rindex" );   
    
    cudaDeviceSynchronize();

    return 0 ; 
}
