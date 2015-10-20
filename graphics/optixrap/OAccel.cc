#include "OAccel.hh"

#include <iostream>
#include <iomanip>
#include <fstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


char* OAccel::read(const char* path)
{
    std::ifstream in( path, std::ifstream::in | std::ifstream::binary );
    if(!in)
    {
        LOG(warning) << "OAccel::read failed for " << path ;
        return NULL ; 
    }

    // Read data from file
    in.seekg (0, std::ios::end);
    std::ifstream::pos_type szp = in.tellg();
    in.seekg (0, std::ios::beg);

    m_size = static_cast<unsigned long long int>(szp);

    if(sizeof(size_t) <= 4 && m_size >= 0x80000000ULL) {
        std::cerr << "[WARNING] acceleration cache file too large for 32-bit application.\n";
        m_loaded = false;
        return NULL ;
    }

    char* data = new char[static_cast<size_t>(m_size)];
    in.read( data, static_cast<std::streamsize>(m_size) );

    return data ; 
}

void OAccel::import()
{
    m_loaded = false;
    char* data = read(m_path);
    if(!data) return ;  

    try {
        m_accel->setData( data, static_cast<RTsize>(m_size) );
        m_loaded = true;
    } 
    catch( optix::Exception& e ) 
    {
        // Setting the acceleration cache failed, but that's not a problem. Since the acceleration
        // is marked dirty and builder and traverser are both set already, it will simply build as usual,
        // without using the cache. So we just warn the user here, but don't do anything else.
        LOG(warning) << "OAccel::import could not use acceleration cache, reason: " << e.getErrorString() ;
    }

    delete[] data;
}



void OAccel::save()
{
    if(m_loaded) return ; 
    if(!m_path) return ;  

    RTsize size = m_accel->getDataSize();
    char* data  = new char[size];
    m_accel->getData( data );

    // Write to file
    LOG(info)<<"OAccel::save " << m_path << " size " << size ;  

    std::ofstream out( m_path, std::ofstream::out | std::ofstream::binary );
    if( !out ) {
      delete[] data;
      std::cerr << "could not open acceleration cache file '" << m_path << "'" << std::endl;
      return;
    }
    out.write( data, size );
    delete[] data;
    std::cerr << "acceleration cache written: '" << m_path << "'" << std::endl;
}


/*
std::string OAccel::getCacheFileName()
{
    std::string cachefile = m_filename;
    size_t idx = cachefile.find_last_of( '.' );
    cachefile.erase( idx );
    cachefile.append( ".accelcache" );
    return cachefile;
}
*/

