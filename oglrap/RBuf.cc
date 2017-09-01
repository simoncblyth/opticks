
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "GLMFormat.hpp"
#include "RBuf.hh"

RBuf::RBuf(unsigned num_items_, unsigned num_bytes_, void* ptr_)
    :
    id(-1),
    num_items(num_items_),
    num_bytes(num_bytes_),
    query_count(-1),
    ptr(ptr_)
{
}

unsigned RBuf::item_bytes() const 
{
    assert( num_bytes % num_items == 0 );
    return num_bytes/num_items ; 
}


RBuf* RBuf::cloneNull() const 
{
    RBuf* b = new RBuf(num_items, num_bytes, NULL) ;
    return b ; 
}

RBuf* RBuf::cloneZero() const 
{
    RBuf* b = new RBuf(num_items, num_bytes, new char[num_bytes]) ;
    memset(b->ptr, 0, num_bytes );
    return b ; 
}




void RBuf::dump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    std::cout << desc() << std::endl ; 
    
    unsigned ib = item_bytes();

    if( ib == sizeof(unsigned) )
    {
        assert( num_items % 3 == 0 );
        unsigned num_tri = num_items/3 ; 
        for(unsigned i=0 ; i < num_tri ; i++ )
        {
             std::cout 
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 0) << " " 
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 1) << " "  
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 2) << " " 
                << std::endl 
                ; 
        }
        std::cout << std::endl ; 
    }
    else if( ib == sizeof(float)*4 )
    {
        for(unsigned i=0 ; i < std::min(num_items,20u) ; i++ ) 
        {        
             const glm::vec4& v = *((glm::vec4*)ptr + i ) ;  
             std::cout << gpresent("v",v) << std::endl ; 
        }
    }
    else if( ib == sizeof(float)*4*4 )
    {
        for(unsigned i=0 ; i < std::min(num_items, 10u) ; i++ ) 
        {        
             const glm::mat4& m = *((glm::mat4*)ptr + i ) ;  
             std::cout << gpresent("m",m) << std::endl ; 
        }
    }
}



std::string RBuf::brief() const 
{
    std::stringstream ss ; 

    //ss << " (" << num_items << "/" << query_count << ") " ; 
    ss << " (" << query_count << ") " ; 

    return ss.str();
}

    
std::string RBuf::desc() const 
{
    std::stringstream ss ; 

    ss << "RBuf"
       << " id " << id  
       << " num_items " << num_items  
       << " num_bytes " << num_bytes
       << " item_bytes() " << item_bytes()
       << " query_count " << query_count
       ; 

    return ss.str();
}

void RBuf::upload(GLenum target, GLenum usage )
{
    glGenBuffers(1, &this->id);
    glBindBuffer(target, this->id);
    glBufferData(target, this->num_bytes, this->ptr, usage);
    glBindBuffer(target, 0);
}

void RBuf::uploadNull(GLenum target, GLenum usage )
{
    glGenBuffers(1, &this->id);
    glBindBuffer(target, this->id);
    glBufferData(target, this->num_bytes, NULL, usage);
    glBindBuffer(target, 0);
}


RBuf* RBuf::Make(const std::vector<glm::vec4>& vert) 
{     
    unsigned num_item = vert.size();
    unsigned num_float = num_item*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, vert.data(), num_byte ) ; 

    return new RBuf( num_item, num_byte, (void*)dest ) ; 
} 

RBuf* RBuf::Make(const std::vector<glm::mat4>& mat) 
{     
    unsigned num_item = mat.size();
    unsigned num_float = num_item*4*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, mat.data(), num_byte ) ; 

    return new RBuf( num_item, num_byte, (void*)dest ) ; 
} 

RBuf* RBuf::Make(const std::vector<unsigned>& elem) 
{     
    unsigned num_item = elem.size();
    unsigned num_unsigned = num_item ; 
    unsigned num_byte = num_unsigned*sizeof(unsigned) ; 

    unsigned* dest = new unsigned[num_unsigned] ; 
    memcpy(dest, elem.data(), num_byte ) ; 

    return new RBuf( num_item, num_byte, (void*)dest ) ; 
} 

 
