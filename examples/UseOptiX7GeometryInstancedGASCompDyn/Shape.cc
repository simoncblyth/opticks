#include <sstream>

#include "Sys.h"
#include "Shape.h"
#include "NP.hh"

unsigned Shape::Type(char typ)  // static 
{
    unsigned t(ZERO) ;  
    switch(typ)
    {
       case 'S': t = SPHERE ; break ; 
       case 'B': t = BOX    ; break ; 
    }
    return t ; 
}

Shape::Shape(const char typ, float sz)
    :
    num(1),
    typs(new char[num]),
    param(new float[4*num]),
    aabb(new float[6*num])
{
    typs[0] = typ ; 
    std::vector<float> szs = { sz } ; 
    init(szs); 
}

Shape::Shape(const char* typs_, const std::vector<float>& szs)
    :
    num(szs.size()),
    typs(new char[num]),
    param(new float[4*num]),
    aabb(new float[6*num])
{
    size_t len = strlen(typs_); 
    for(unsigned i=0 ; i < num ; i++) typs[i] = i < len ? typs_[i] : typs_[0] ;    // duplicate the first typ, if run out 
    init(szs); 
}

Shape::~Shape()
{
    delete [] typs ; 
    delete [] param ; 
    delete [] aabb ; 
}

void Shape::init( const std::vector<float>& szs )
{
    for(unsigned i=0 ; i < szs.size() ; i++) assert(szs[i] <= szs[0] ) ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        float size = szs[i] ;  
        char type = typs[i] ; 

        param[0+4*i] = size ; 
        param[1+4*i] = 0.f ; 
        param[2+4*i] = 0.f ; 
        param[3+4*i] = Sys::unsigned_as_float(type) ; 

        aabb[0+6*i] = -size ; 
        aabb[1+6*i] = -size ; 
        aabb[2+6*i] = -size ; 
        aabb[3+6*i] =  size ; 
        aabb[4+6*i] =  size ; 
        aabb[5+6*i] =  size ; 
    }
}


float* Shape::get_aabb(unsigned idx) const
{
    assert( idx < num ); 
    return aabb + idx*6 ; 
}
float* Shape::get_param(unsigned idx) const
{
    assert( idx < num ); 
    return param + idx*4 ; 
}
char Shape::get_type(unsigned idx) const
{
    assert( idx < num ); 
    return typs[idx] ; 
}
float Shape::get_size(unsigned idx) const
{
    assert( idx < num ); 
    return param[0+4*idx] ; 
}

std::string Shape::desc(unsigned idx) const 
{  
    std::stringstream ss ; 
    ss << " idx: " << idx  ;
    ss << " typ: " << get_type(idx)  ;
    ss << " param: " ; 
    for(unsigned i=0 ; i < 4 ; i++) ss << param[i+4*idx] << " "  ; 
    ss << " aabb: " ; 
    for(unsigned i=0 ; i < 6 ; i++) ss << aabb[i+6*idx] << " "  ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string Shape::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++ ) ss << desc(i) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void Shape::write(const char* base, const char* rel, unsigned idx) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel << "/" << idx << "/" ; 
    std::string dir = ss.str();   
    NP::Write(dir.c_str(), "aabb.npy",   aabb,  num, 2, 3 ); 
    NP::Write(dir.c_str(), "param.npy",  param, num, 1, 4 ); 
}


