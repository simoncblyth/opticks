
#include <iostream>
#include <csignal>

#include "PLOG.hh"
#include "NPY.hpp"
#include "BStr.hh"
#include "GItemList.hh"

#include "GPt.hh"
#include "GPts.hh"


#include "PLOG.hh"

const plog::Severity GPts::LEVEL = PLOG::EnvLevel("GPts", "DEBUG") ; 


const char* GPts::GPTS_LIST = "GPts" ; 

GPts* GPts::Make()  // static
{
    LOG(LEVEL) ; 
    //std::raise(SIGINT); 
    NPY<int>* ipt = NPY<int>::make(0, 4 ) ; 
    NPY<float>* plc = NPY<float>::make(0, 4, 4 ) ; 
    GItemList* specs = new GItemList(GPTS_LIST, "") ; 
    return new GPts(ipt, plc, specs); 
}

GPts* GPts::Load(const char* dir)  // static
{
    LOG(LEVEL) << dir ; 
    NPY<int>* ipt = LoadBuffer<int>(dir, "ipt") ; 
    NPY<float>* plc = LoadBuffer<float>(dir, "plc"); 
    GItemList* specs = GItemList::Load(dir, GPTS_LIST, "") ; 
    GPts* pts = new GPts(ipt, plc, specs); 
    pts->import(); 
    return pts ; 
}

void GPts::save(const char* dir)
{
    LOG(LEVEL) << dir ; 
    export_();  
    if(m_ipt_buffer) m_ipt_buffer->save(dir, BufferName("ipt"));    
    if(m_plc_buffer) m_plc_buffer->save(dir, BufferName("plc"));    
    if(m_specs) m_specs->save(dir); 
}


template<typename T>
NPY<T>* GPts::LoadBuffer(const char* dir, const char* tag) // static
{
    const char* name = BufferName(tag) ;
    bool quietly = true ; 
    NPY<T>* buf = NPY<T>::load(dir, name, quietly ) ;
    return buf ; 
}

const char* GPts::BufferName(const char* tag) // static
{
    return BStr::concat(tag, "Buffer.npy", NULL) ;
}

GPts::GPts(NPY<int>* ipt, NPY<float>* plc, GItemList* specs) 
    :
    m_ipt_buffer(ipt),
    m_plc_buffer(plc),
    m_specs(specs)
{
}

void GPts::export_() // to the buffer
{
    for(unsigned i=0 ; i < getNumPt() ; i++ )
    {
        const GPt* pt = getPt(i); 
        glm::ivec4 ipt(pt->lvIdx, pt->ndIdx, pt->csgIdx, i); 

        m_specs->add(pt->spec.c_str());
        m_ipt_buffer->add(ipt); 
        m_plc_buffer->add(pt->placement) ;  
    }
}
 
void GPts::import()  // from buffers into vector
{
    assert( getNumPt() == 0 );  

    unsigned num_pt = m_specs->getNumItems(); 
    assert( num_pt == m_ipt_buffer->getShape(0)) ; 
    assert( num_pt == m_plc_buffer->getShape(0)) ; 

    for(unsigned i=0 ; i < num_pt ; i++)
    {
        const char* spec = m_specs->getKey(i); 
        glm::mat4 placement = m_plc_buffer->getMat4(i); 
        glm::ivec4 ipt = m_ipt_buffer->getQuadI(i); 
  
        GPt* pt = new GPt( ipt.x, ipt.y, ipt.z, spec, placement ); 
        add(pt);  
    }
    assert( getNumPt() == num_pt );  
    LOG(LEVEL) << " num_pt " << num_pt ; 
}


unsigned GPts::getNumPt() const { return m_pts.size() ; } 

const GPt* GPts::getPt(unsigned i) const 
{
    assert( i < m_pts.size() ); 
    return m_pts[i] ; 
}

void GPts::add( GPt* other )
{
    m_pts.push_back(other);    
}

std::string GPts::brief() const 
{
    std::stringstream ss ; 

    unsigned num_pt = getNumPt() ; 
    ss << " GPts.NumPt " << num_pt
       << " lvIdx (" 
        ;

    for( unsigned i=0 ; i < num_pt ; i++) 
    {
        const GPt* pt = getPt(i); 
        ss << " " << pt->lvIdx  ; 
    } 
    ss << ")" ; 

    return ss.str(); 
}


void GPts::dump(const char* msg) const 
{
    LOG(info) << msg << brief() ; 
    for(unsigned i=0 ; i < getNumPt() ; i++ )
    {
        const GPt* pt = getPt(i); 
        std::cout 
            << " i " << std::setw(4) << i 
            << pt->desc()
            << std::endl 
            ; 
    }
}


template GGEO_API NPY<float>* GPts::LoadBuffer<float>(const char*, const char*) ;
template GGEO_API NPY<int>* GPts::LoadBuffer<int>(const char*, const char*) ;
template GGEO_API NPY<unsigned>* GPts::LoadBuffer<unsigned>(const char*, const char*) ;


