#include <cassert>
#include <sstream>
#include <cfloat>
#include <cstring>


#include "NPY.hpp"

#include "GLMPrint.hpp"
#include "NGLM.hpp"

#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"

#include "PLOG.hh"

const char* ViewNPY::BYTE_ = "BYTE"; 
const char* ViewNPY::UNSIGNED_BYTE_ = "UNSIGNED_BYTE" ; 
const char* ViewNPY::SHORT_ = "SHORT"; 
const char* ViewNPY::UNSIGNED_SHORT_ = "UNSIGNED_SHORT" ; 
const char* ViewNPY::INT_ = "INT" ; 
const char* ViewNPY::UNSIGNED_INT_ = "UNSIGNED_INT" ; 
const char* ViewNPY::HALF_FLOAT_ = "HALF_FLOAT" ; 
const char* ViewNPY::FLOAT_ = "FLOAT" ; 
const char* ViewNPY::DOUBLE_ = "DOUBLE" ; 
const char* ViewNPY::FIXED_ = "FIXED" ; 
const char* ViewNPY::INT_2_10_10_10_REV_ = "INT_2_10_10_10_REV"  ; 
const char* ViewNPY::UNSIGNED_INT_2_10_10_10_REV_ = "UNSIGNED_INT_2_10_10_10_REV" ; 
const char* ViewNPY::UNSIGNED_INT_10F_11F_11F_REV_ = "UNSIGNED_INT_10F_11F_11F_REV" ;

const char* ViewNPY::getTypeName()
{
    const char* name = NULL ; 
    switch(m_type)
    {
       case BYTE:           name = BYTE_            ;break;
       case UNSIGNED_BYTE:  name = UNSIGNED_BYTE_   ;break;
       case SHORT:          name = SHORT_           ;break;
       case UNSIGNED_SHORT: name = UNSIGNED_SHORT_  ;break;
       case INT:            name = INT_             ;break;
       case UNSIGNED_INT:   name = UNSIGNED_INT_    ;break;
       case HALF_FLOAT:     name = HALF_FLOAT_      ;break;
       case FLOAT:          name = FLOAT_           ;break;
       case DOUBLE:         name = DOUBLE_          ;break;
       case FIXED:          name = FIXED_           ;break;
       case INT_2_10_10_10_REV: name = INT_2_10_10_10_REV_  ;break;
       case UNSIGNED_INT_2_10_10_10_REV: name = UNSIGNED_INT_2_10_10_10_REV_ ;break;
       case UNSIGNED_INT_10F_11F_11F_REV: name = UNSIGNED_INT_10F_11F_11F_REV_ ;break;
    }
    return name ;
}





ViewNPY::ViewNPY(const char* name, NPYBase* npy, unsigned int j, unsigned int k, unsigned int l, unsigned int size, Type_t type, bool norm, bool iatt, unsigned int item_from_dim) 
  :
            m_name(strdup(name)),
            m_npy(npy),
            m_parent(NULL),
            m_bytes(NULL),
            m_j(j),
            m_k(k),
            m_l(l),
            m_size(size),
            m_type(type),
            m_norm(norm),
            m_iatt(iatt),
            m_item_from_dim(item_from_dim),

            m_numbytes(0),
            m_stride(0),
            m_offset(0),
            m_low(NULL),
            m_high(NULL),
            m_dimensions(NULL),
            m_center(NULL),
            m_model_to_world(0),
            m_center_extent(0),
            m_extent(0.f),
            m_addressed(false)
{
    init();
}


glm::vec4& ViewNPY::getCenterExtent()
{
    return m_center_extent ; 
} 
glm::mat4& ViewNPY::getModelToWorld()
{
    return m_model_to_world ; 
}
float* ViewNPY::getModelToWorldPtr()
{
    return glm::value_ptr(m_model_to_world) ; 
}
float ViewNPY::getExtent()
{
    return m_extent ; 
}



MultiViewNPY* ViewNPY::getParent()
{
    return m_parent ; 
}
void ViewNPY::setParent(MultiViewNPY* parent)
{
    m_parent = parent ; 
}




NPYBase*     ViewNPY::getNPY(){    return m_npy   ; }
void*        ViewNPY::getBytes(){  return m_bytes ; }
unsigned int ViewNPY::getNumBytes(){  return m_numbytes ; }
unsigned int ViewNPY::getStride(){ return m_stride ; }
unsigned long ViewNPY::getOffset(){ return m_offset ; }
unsigned int ViewNPY::getSize(){   return m_size ; }  //typically 1,2,3,4 
bool         ViewNPY::getNorm(){ return m_norm ; }
bool         ViewNPY::getIatt(){ return m_iatt ; }
ViewNPY::Type_t  ViewNPY::getType(){ return m_type ; }
const char*  ViewNPY::getName(){ return m_name ; }
 

std::string ViewNPY::getShapeString()
{
    return m_npy->getShapeString();
}
unsigned int ViewNPY::getNumQuads()
{
    return m_npy->getNumQuads();
}

void ViewNPY::init()
{
    assert(m_npy);

    m_bytes    = m_npy->getBytes() ;

    assert(m_item_from_dim == 1 || m_item_from_dim == 2);

    // these dont require the data, just the shape
    m_numbytes = m_npy->getNumBytes(0) ;
    m_stride   = m_npy->getNumBytes(m_item_from_dim) ;
    m_offset   = m_npy->getByteIndex(0,m_j,m_k,m_l) ;  //  i*nj*nk*nl + j*nk*nl + k*nl + l     scaled by sizeoftype

    if( m_npy->hasData() )
    { 
        addressNPY();
    } 
}

unsigned int ViewNPY::getCount()
{
    unsigned int count(0) ;

    if(m_item_from_dim == 1)   // the default, only 1st dim is count
    {
        count =  m_npy->getShape(0) ;
    }
    else if(m_item_from_dim == 2)   // structured records, 1st*2nd dim is count
    {
        count =  m_npy->getShape(0)*m_npy->getShape(1) ;
    }
    else 
    {
        assert(0 && "bad m_item_from_dim");
    }               

    if(count == 0)
    {

        const char* bufname = m_npy->getBufferName();

        if(bufname && strcmp(bufname, "nopstep")==0)
        {
           LOG(debug) << "ViewNPY::getCount UNEXPECTED"
                        << " bufname " << bufname
                        << " desc " << description()
                        << " count " << count 
                        << " shape " <<  getShapeString() 
                        ;
        }
        else
        {
           LOG(warning) << "ViewNPY::getCount UNEXPECTED"
                        << " bufname " << bufname
                        << " desc " << description()
                        << " count " << count 
                        << " shape " <<  getShapeString() 
                        ;
        }
    }

    return count ; 
}

void ViewNPY::addressNPY()
{
    m_addressed = true ; 
    findBounds();
}

unsigned int ViewNPY::getValueOffset()
{
    //   i*nj*nk + j*nk + k ;    i=0
    //
    // serial offset of the qty within each rec 
    // obtained from first rec (i=0)
    //
    return m_npy->getValueIndex(0,m_j,m_k,m_l); 
}

void ViewNPY::setCustomOffset(unsigned long offset)
{
    m_offset = offset ;
}



void ViewNPY::dump(const char* msg)
{
    float xx[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};
    float yy[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};
    float zz[4] = { FLT_MAX, -FLT_MAX, 0.f, 0.f};

    printf("%s name %s \n", msg, m_name);
    const char* fmt = "ViewNPY::dump %5s %6u/%6u :  %15f %15f %15f \n";


    unsigned int count = getCount();

    for(unsigned int i=0 ; i < count ; ++i )
    {   
        char* ptr = (char*)m_bytes + m_offset + i*m_stride  ;   
        float* f = (float*)ptr ; 
        float x(*(f+0));
        float y(*(f+1));
        float z(*(f+2));

        if( x<xx[0] ) xx[0] = x ;  
        if( x>xx[1] ) xx[1] = x ;  

        if( y<yy[0] ) yy[0] = y ;  
        if( y>yy[1] ) yy[1] = y ;  

        if( z<zz[0] ) zz[0] = z ;  
        if( z>zz[1] ) zz[1] = z ;  

        if(i < 5 || i > count - 5) printf(fmt, "", i,count, x, y, z);
    }

    xx[2] = xx[1] - xx[0] ;
    yy[2] = yy[1] - yy[0] ;
    zz[2] = zz[1] - zz[0] ;

    xx[3] = (xx[1] + xx[0])/2.f ;
    yy[3] = (yy[1] + yy[0])/2.f ;
    zz[3] = (zz[1] + zz[0])/2.f ;

    printf(fmt, "min", 0,0,xx[0],yy[0],zz[0]);
    printf(fmt, "max", 0,0,xx[1],yy[1],zz[1]);
    printf(fmt, "dif", 0,0,xx[2],yy[2],zz[2]);
    printf(fmt, "cen", 0,0,xx[3],yy[3],zz[3]);
}


void ViewNPY::findBounds()
{
    if(strcmp(m_name, "rsel") == 0)
    {
        LOG(warning) << "ViewNPY::findBounds skipping for " << m_name  ;
        return ; 
    }

    glm::vec3 lo( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    //printf("ViewNPY::findBounds name %s bytes %p offset %lu stride %u count %u \n", m_name, m_bytes, m_offset, m_stride, m_count );

    unsigned int count = getCount();
    for(unsigned int i=0 ; i < count ; ++i )
    {   
        char* ptr = (char*)m_bytes + m_offset + i*m_stride  ;   
        float* f = (float*)ptr ; 

        glm::vec3 v(*(f+0),*(f+1),*(f+2));

        lo.x = std::min( lo.x, v.x);
        lo.y = std::min( lo.y, v.y);
        lo.z = std::min( lo.z, v.z);

        hi.x = std::max( hi.x, v.x);
        hi.y = std::max( hi.y, v.y);
        hi.z = std::max( hi.z, v.z);

    }

    m_low = new glm::vec3(lo.x, lo.y, lo.z);
    m_high = new glm::vec3(hi.x, hi.y, hi.z);

    m_dimensions = new glm::vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z );
    m_center     = new glm::vec3((hi.x + lo.x)/2.0f, (hi.y + lo.y)/2.0f , (hi.z + lo.z)/2.0f );

    m_extent = 0.f ;
    m_extent = std::max( m_dimensions->x , m_extent );
    m_extent = std::max( m_dimensions->y , m_extent );
    m_extent = std::max( m_dimensions->z , m_extent );
    m_extent = m_extent / 2.0f ;    

    if(m_extent < 1.f )
    {
        LOG(debug) << "ViewNPY::findBounds setting nominal extent as auto-extent too small : " << m_extent ; 

        // for Torch gensteps with all steps at same position, the extent comes out as zero 
        // resulting in a blank render
        //
        //print( *m_low, "m_low");
        //print( *m_high, "m_high");
        //print( *m_dimensions, "m_dimensions");
        //print( *m_center, "m_center");

        m_extent = 1000.f ; 
    }


    glm::vec3 s(m_extent);
    glm::vec3 t(*m_center);
    m_model_to_world = glm::scale(glm::translate(glm::mat4(1.0), t), s); 

    m_center_extent.x = (hi.x + lo.x)/2.0f ;
    m_center_extent.y = (hi.y + lo.y)/2.0f ;
    m_center_extent.z = (hi.z + lo.z)/2.0f ;
    m_center_extent.w = m_extent ; 

    //Summary("ViewNPY::findBounds");
}
void ViewNPY::Summary(const char* msg)
{
    Print(msg);

    if(!m_low) return ;

    print(*m_low,  "m_low");
    print(*m_high, "m_high");
    print(*m_dimensions, "m_dimensions");
    print(*m_center,     "m_center");
    print(m_model_to_world, "m_model_to_world");
    print(glm::value_ptr(m_model_to_world), "glm::value_ptr(m_model_to_world)");
}


void ViewNPY::Print(const char* msg)
{
    unsigned int count = getCount();
    printf("%s name %s type [%d] typeName %s numbytes %u stride %u offset %lu count %u extent %f\n", msg, m_name, m_type, getTypeName(), m_numbytes, m_stride, m_offset, count, m_extent );
}

std::string ViewNPY::description()
{
    std::stringstream ss ;
    ss << "ViewNPY " 
       << ( m_parent ? m_parent->getName() : "" )
       <<  " " << std::setw(10) << m_name ;

    return ss.str();
}




