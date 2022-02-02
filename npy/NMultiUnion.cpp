#include <limits>

#include "NBBox.hpp"
#include "nmat4triple.hpp"
#include "NMultiUnion.hpp"

nbbox nmultiunion::bbox() const 
{
    std::cout << "nmultiunion::bbox subs.size " << subs.size() << std::endl ; 

    nbbox bb = make_bbox() ; 

    for(unsigned isub=0 ; isub < subs.size() ; isub++)
    {
        const nnode* sub = subs[isub] ; 

        std::cout 
            << " isub " << std::setw(5) << isub 
            << " sub->gtransform " << std::setw(10) << sub->gtransform
            << " sub->transform " << std::setw(10) << sub->transform
            << std::endl 
            ;

        nbbox sub_bb = sub->bbox();  
        sub_bb.dump(); 

        bb.include(sub_bb); 
    }

    // gtransform is the composite one
    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}



float nmultiunion::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    float sd = std::numeric_limits<float>::max() ;  

    for(unsigned isub=0 ; isub < subs.size() ; isub++)
    {
        const nnode* sub = subs[isub] ; 
        float sd_sub = (*sub)( p.x, p.y, p.z );  
        sd = std::min( sd, sd_sub );   
    }

    return complement ? -sd : sd ;
} 






int nmultiunion::par_euler() const 
{
    return 0 ;  
}

unsigned nmultiunion::par_nsurf() const 
{
    return 0 ;  
}
unsigned nmultiunion::par_nvertices(unsigned , unsigned ) const 
{
    return 0 ;  
}


