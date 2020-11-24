
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>

#include <cuda_runtime_api.h>


#include "CDevice.hh"
#include "LaunchCommon.hh"   // mkdirp
#include "PLOG.hh"

const plog::Severity CDevice::LEVEL = PLOG::EnvLevel("CDevice", "DEBUG"); 


const char* CDevice::CVD = "CUDA_VISIBLE_DEVICES" ; 

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

Use CUDA API to collect a summary of the cudaDeviceProp properties 
regarding all attached devices into the vector of CDevice argument.

When ordinal_from_index=true the CDevice.ordinal value is taken 
from the index in the order returned by cudaGetDeviceProperties(&p, i)

**/

void CDevice::Collect(std::vector<CDevice>& devices, bool ordinal_from_index)
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    LOG(LEVEL) << "cudaGetDeviceCount : " << devCount ; 

    for (int i = 0; i < devCount; ++i)
    {   
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i); 

        CDevice d ;   

        assert( sizeof(p.name) == sizeof(char)*256 ) ;  
        strncpy( d.name, p.name, sizeof(p.name) ); 

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 10000)
        assert( sizeof(p.uuid) == sizeof(uuid) ); 
        strncpy( d.uuid, p.uuid.bytes, sizeof(p.uuid) ); 
#elif (CUDART_VERSION >= 9000)
#endif

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
        sizeof(int) +   // ordinal
        sizeof(int) +   // index
        sizeof(char)*256 +  // name 
        sizeof(char)*16 +     // uuid
        sizeof(int) +     // major 
        sizeof(int) +     // minor 
        sizeof(int) +   // compute_capability
        sizeof(int) +   // multiProcessorCount 
        sizeof(size_t) ;   // totalGlobalMem
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
    assert( p - buffer == size ); 
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



/**
CDevice::Visible
------------------

This assumes that the ordinal is the index when all GPUs are visible 
and it finds this by arranging to persist the query when 
CUDA_VISIBLE_DEVICES is not defined and use that to provide something 
to match against when the envvar is defined.

Initially tried to do this in one go by changing envvar 
and repeating the query. But that doesnt work, 
presumably as the CUDA_VISIBLE_DEVICES value only has 
any effect when cuda is initialized.

Of course the disadvantage of this approach 
is that need to arrange to do the persisting of all devices 
at some initialization time and need to find an 
appropriate place for the file.

The purpose is for reference running, especially performance
scanning : so its acceptable to require running a metadata
capturing executable prior to scanning.

Possibly NVML can provide a better solution, see nvml-
Actually maybe not : the NVML enumeration order follows nvidia-smi 
not CUDA. 

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
        LOG(LEVEL) << " no_cvd " ; 
        if(!nosave)
        Save( visible, dirpath );      
    }
    else
    {
        LOG(LEVEL) << " with cvd " << cvd ; 
        Load(all,  dirpath); 

        for(unsigned i=0 ; i < visible.size() ; i++)
        {
            CDevice& v = visible[i] ; 
            v.ordinal = FindIndexOfMatchingDevice( v, all );   
        }
    }
}

/**
CDevice::FindIndexOfMatchingDevice
------------------------------------

**/

int CDevice::FindIndexOfMatchingDevice( const CDevice& d, const std::vector<CDevice>& all )
{
    int index = -1 ; 
    LOG(LEVEL) 
         << " d " << d.desc() 
         << " all.size " << all.size()
         ;  

    for(unsigned i=0 ; i < all.size() ; i++)
    {
        const CDevice& a = all[i] ; 
        bool m = a.matches(d) ; 
        LOG(LEVEL) 
            << " a " << a.desc()
            << " m " << m 
            ;  

        if(m)
        {
           index = a.index ; 
           break ; 
        } 
    }
    LOG(LEVEL) << " index : " << index ;  
    return index ; 
}


void CDevice::Dump( const std::vector<CDevice>& devices, const char* msg )
{
    LOG(info) << msg << "[" << Brief(devices) << "]" ; 
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


void CDevice::PrepDir(const char* dirpath)
{
    mkdirp(dirpath, 0777);
}

void CDevice::Save( const std::vector<CDevice>& devices, const char* dirpath)
{
    std::string path = Path(dirpath); 
    PrepDir(dirpath); 
    LOG(LEVEL) << "path " << path ; 

    std::ofstream out(path.c_str(), std::ofstream::binary);
    if(out.fail())
    {
        LOG(error) << " failed open for " << path ; 
        return ; 
    }

    for(unsigned i = 0 ; i < devices.size() ; ++i )
    {
        const CDevice& d = devices[i] ; 
        d.write(out);   
    }
}

void CDevice::Load( std::vector<CDevice>& devices, const char* dirpath)
{
    std::string path = Path(dirpath); 
    LOG(LEVEL) 
        << "dirpath " << dirpath 
        << "path " << path 
        ; 
    std::ifstream in(path.c_str(), std::ofstream::binary);

    CDevice d ; 
    while(true)
    {
        d.read(in);   
        if(in.eof()) return ;   
        if(in.fail())
        {
            LOG(error) << " failed read from " << path ; 
            return ; 
        } 
        devices.push_back(d); 
    }
}

std::string CDevice::Brief( const std::vector<CDevice>& devices )
{ 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < devices.size() ; i++)
    {
        const CDevice& d = devices[i] ; 
        ss << d.ordinal << ':' ; 
        for(unsigned j=0 ; j < strlen(d.name) ; j++)
        {   
            char c = *(d.name+j) ;   
            ss << ( c == ' ' ? '_' : c ) ;   
        }
        if( i < devices.size() - 1 ) ss << ' ' ;
    }   
    return ss.str(); 
}


