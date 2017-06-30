
#include "GLMFormat.hpp"
#include "PLOG.hh"

#include "NNodeEnum.hpp"
#include "NSDF.hpp"

NSDF::NSDF(std::function<float(float,float,float)> sdf, const glm::mat4& inverse )
        :
        sdf(sdf),
        inverse(inverse),
        tot(0,0,0,0),
        range(0,0),
        epsilon(0),
        expect(0),
        qqptr(NULL)
    {
    }
    
float NSDF::operator()( const glm::vec3& q_ )
{
    glm::vec4 q(q_,1.0); 
    q = inverse * q ; 
    float sd = sdf(q.x, q.y, q.z);

/*
    std::cout 
        << " q_ " << gpresent(q_) 
        << " q "  << gpresent(q) 
        << " sd " << std::scientific << sd 
        << std::endl 
        ;

*/
    return sd ; 
}


void NSDF::clear()
{
    tot.x = 0 ; 
    tot.y = 0 ; 
    tot.z = 0 ; 
    tot.w = 0 ; 

    range.x = 0 ; 
    range.y = 0 ; 

    epsilon = 0 ; 
    expect = 0 ; 
    qqptr = NULL ;

    sd.clear();
}

void NSDF::classify(const std::vector<glm::vec3>& qq, float epsilon, unsigned expect )
{
    clear();

    this->qqptr = &qq ; 
    this->epsilon = epsilon ; 
    this->expect = expect ; 

    std::transform( qq.begin(), qq.end(), std::back_inserter(sd), *this );

    if(sd.size() > 0)
    {
        std::pair<VFI, VFI> p = std::minmax_element(sd.begin(), sd.end());
        range.x = *p.first ; 
        range.y = *p.second ; 
    }
    else
    {
        LOG(warning) << " sd.size ZERO " ; 
    }

    assert( qq.size() == sd.size());

    for(unsigned i=0 ; i < sd.size() ; i++)
    {  
        NNodePointType pt = NNodeEnum::PointClassify(sd[i], epsilon ); 
        switch(pt)
        {
            case POINT_INSIDE : tot.x++ ; break ; 
            case POINT_SURFACE: tot.y++ ; break ; 
            case POINT_OUTSIDE: tot.z++ ; break ; 
        }
        if(pt & ~expect) tot.w++ ; 

/*
        std::cout << "NSDF::classify" 
                  << " i " << std::setw(4) << i 
                  << " q " << qq[i]
                  ;
*/

    }
} 

bool NSDF::is_error() const { return tot.w > 0 ; }
bool NSDF::is_empty() const { return sd.size() == 0 ; }

std::string NSDF::desc() const 
{
    std::stringstream ss ; 
    ss 
       << ( is_error() ? "EE" : "  " ) 
       << ( is_empty() ? "??" : "  " ) 
       << std::setw(4) << sd.size()
       << "("
       << "in" << ( expect & POINT_INSIDE ? ":" : ""  )
       << "/"
       << "su" << ( expect & POINT_SURFACE ? ":" : "" )
       << "/"
       << "ou" << ( expect & POINT_OUTSIDE ? ":" : "" )
       << "/"
       << "er"
       << ") "
       << gpresent( tot, 3 )
       << " "
       << gpresent( range )
       ;
    return ss.str();
}


std::string NSDF::detail() const 
{
    std::stringstream ss ; 
    ss 
       << " ep " 
       << std::scientific << epsilon
       << " [" 
       << std::scientific << range[0]
       << "," 
       << std::scientific << range[1] 
       << "]" 
       ;
    return ss.str();
}





