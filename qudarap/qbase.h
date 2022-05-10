#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QBASE_METHOD __device__
#else
   #define QBASE_METHOD 
#endif 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <sstream>
#include <string>
#endif 


struct qbase
{
    int pidx ; 
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    QBASE_METHOD std::string desc() const ; 
#endif
};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline QBASE_METHOD std::string qbase::desc() const 
{
    std::stringstream ss ; 
    ss << "qbase::desc"
       << " pidx " << pidx 
       ; 
    std::string s = ss.str(); 
    return s ; 
}
#endif




