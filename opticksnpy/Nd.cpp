#include <iomanip>
#include <sstream>

#include "Nd.hpp"
#include "NGLMExt.hpp"


std::string nd::desc()
{
    std::stringstream ss ; 

    ss << "nd"
       << " [" 
       << std::setw(3) << idx 
       << ":" 
       << std::setw(3) << mesh 
       << ":" 
       << std::setw(3) << children.size() 
       << ":" 
       << std::setw(2) << depth
       << "]"
       << " " << boundary 
       ;

    return ss.str();
}

nmat4triple* nd::make_global_transform(nd* n)
{
    std::vector<nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, true) ; 
}


