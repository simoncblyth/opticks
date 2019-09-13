
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>

#include "CDevice.hh"
#include "PLOG.hh"

const char* CDevice::desc() const 
{
    std::stringstream ss ; 
    // uuid is not printable
    ss << "CDevice"
       << " index " << index
       << " ordinal " << ordinal
       << " name " << name
       << " major " << major 
       << " minor " << minor 
       << " compute_capability " << compute_capability 
       << " multiProcessorCount " << multiProcessorCount
       << " totalGlobalMem " << totalGlobalMem
       ; 
    std::string s = ss.str(); 
    return strdup(s.c_str());  
} 

bool CDevice::matches(const CDevice& other) const 
{
   return strncmp(other.uuid, uuid, sizeof(uuid)) == 0 && strncmp(other.name, name, sizeof(name)) == 0;   
}



/**
CDevice::Collect
--------------------


**/

void CDevice::Collect(std::vector<CDevice>& devices, bool ordinal_from_index)
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    LOG(info) << "devCount: " << devCount ; 

    for (int i = 0; i < devCount; ++i)
    {   
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i); 

        CDevice d ;   

        assert( sizeof(p.name) == sizeof(name) ) ;  
        strncpy( d.name, p.name, sizeof(p.name) ); 

        assert( sizeof(p.uuid) == sizeof(uuid) ); 
        strncpy( d.uuid, p.uuid.bytes, sizeof(p.uuid) ); 

        d.index = i ; 
        d.ordinal = ordinal_from_index ? i : -1 ;    
        d.major = p.major ; 
        d.minor = p.minor ; 
        d.compute_capability = p.major*10 + p.minor ; 

        d.multiProcessorCount = p.multiProcessorCount ;  
        d.totalGlobalMem = p.totalGlobalMem ; 

        devices.push_back(d); 
    }   
}

int CDevice::Size() 
{
    return 
        sizeof(ordinal) + 
        sizeof(index) + 
        sizeof(name) + 
        sizeof(uuid) + 
        sizeof(major) + 
        sizeof(minor) + 
        sizeof(compute_capability) + 
        sizeof(multiProcessorCount) + 
        sizeof(totalGlobalMem) ;  
}
void CDevice::write( std::ostream& out ) const
{
    int size = Size(); 
    char* buffer = new char[size];
    char* p = buffer ; 

    memcpy( p, &ordinal,             sizeof(ordinal) )             ; p += sizeof(ordinal) ; 
    memcpy( p, &index,               sizeof(index) )               ; p += sizeof(index) ; 
    memcpy( p, name,                 sizeof(name) )                ; p += sizeof(name) ; 
    memcpy( p, uuid,                 sizeof(uuid) )                ; p += sizeof(uuid) ; 
    memcpy( p, &major,               sizeof(major) )               ; p += sizeof(major) ; 
    memcpy( p, &minor,               sizeof(minor) )               ; p += sizeof(minor) ; 
    memcpy( p, &compute_capability,  sizeof(compute_capability) )  ; p += sizeof(compute_capability) ; 
    memcpy( p, &multiProcessorCount, sizeof(multiProcessorCount) ) ; p += sizeof(multiProcessorCount) ; 
    memcpy( p, &totalGlobalMem,      sizeof(totalGlobalMem) )      ; p += sizeof(totalGlobalMem) ; 

    out.write(buffer, size);   
    delete [] buffer ; 
}

void CDevice::read( std::istream& in )
{
    int size = Size(); 
    char* buffer = new char[size];
    in.read(buffer, size);   
    char* p = buffer ; 

    memcpy( &ordinal,  p,           sizeof(ordinal) )             ; p += sizeof(ordinal) ; 
    memcpy( &index,    p,           sizeof(index) )               ; p += sizeof(index) ; 
    memcpy( name,      p,           sizeof(name) )                ; p += sizeof(name) ; 
    memcpy( uuid,      p,           sizeof(uuid) )                ; p += sizeof(uuid) ; 
    memcpy( &major,    p,           sizeof(major) )               ; p += sizeof(major) ; 
    memcpy( &minor,    p,           sizeof(minor) )               ; p += sizeof(minor) ; 
    memcpy( &compute_capability, p, sizeof(compute_capability) )  ; p += sizeof(compute_capability) ; 
    memcpy( &multiProcessorCount,p, sizeof(multiProcessorCount) ) ; p += sizeof(multiProcessorCount) ; 
    memcpy( &totalGlobalMem,     p, sizeof(totalGlobalMem) )      ; p += sizeof(totalGlobalMem) ; 

    delete [] buffer ; 
}  

const char* CDevice::CVD = "CUDA_VISIBLE_DEVICES" ; 


/**
CDevice::Visible
------------------

This assumes that the ordinal is the index when all GPUs are are visible 
and it finds this by making two queries when CUDA_VISIBLE_DEVICES is
defined, one with the envvar removed and one with it restored.

But this doesnt work, presumably as only the CUDA_VISIBLE_DEVICES
value when cuda initializes has any effect.


**/

void CDevice::Visible(std::vector<CDevice>& visible, const char* dirpath, bool nosave)
{
    char* cvd = getenv(CVD); 
    bool no_cvd = cvd == NULL ;  
    std::vector<CDevice> all ; 

    bool ordinal_from_index = no_cvd  ; 
    Collect(visible, ordinal_from_index); 

    if( no_cvd )
    {
        LOG(info) << " no_cvd " ; 
        if(!nosave)
        Save( visible, dirpath );      
    }
    else
    {
        LOG(info) << " with cvd " << cvd ; 
        Load( all, dirpath );
        LOG(info) << " load all " << all.size() ; 
    }

    for(unsigned i=0 ; i < visible.size() ; i++)
    {
        CDevice& v = visible[i] ; 
        v.ordinal = FindIndexOfMatchingDevice( v, all );   
    }
}

int CDevice::FindIndexOfMatchingDevice( const CDevice& d, const std::vector<CDevice>& all )
{
    int index = -1 ; 
    for(unsigned i=0 ; i < all.size() ; i++)
    {
        const CDevice& a = all[i] ; 
        if( a.matches(d) )
        {
           index = a.index ; 
           break ; 
        } 
    }
    return index ; 
}


void CDevice::Dump( const std::vector<CDevice>& devices)
{
    for(unsigned i=0 ; i < devices.size() ; i++)
    {
        const CDevice& d = devices[i] ; 
        LOG(info) << d.desc();    
    }  
}


const char* CDevice::FILENAME = "CDevice.bin" ; 

std::string CDevice::Path(const char* dirpath)
{
    std::stringstream ss ; 
    if( dirpath ) ss << dirpath << "/" ; 
    ss << FILENAME ;  
    return ss.str(); 
}


void CDevice::Save( const std::vector<CDevice>& devices, const char* dirpath)
{
    std::string path = Path(dirpath); 
    std::ofstream out(path.c_str(), std::ofstream::binary);
    for(unsigned i = 0 ; i < devices.size() ; ++i )
    {
        const CDevice& d = devices[i] ; 
        d.write(out);   
    }
}

void CDevice::Load( std::vector<CDevice>& devices, const char* dirpath)
{
    std::string path = Path(dirpath); 
    std::ifstream in(path.c_str(), std::ofstream::binary);
    CDevice d ; 
    while(true)
    {
        d.read(in);   
        if(in.eof()) return ;   
        devices.push_back(d); 
    }
}


