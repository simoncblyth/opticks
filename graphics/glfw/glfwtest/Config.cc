#include "Config.hh"
#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>
#include <iostream>

namespace po = boost::program_options;

Config::Config()
    : 
    ConfigBase(),
    m_zmq_backend_default("tcp://127.0.0.1:5002"),
    m_udp_port(8080),
    m_udp_port_default(8080),
    m_yfov(60.f),
    m_yfov_default(60.f)
{
    create_descriptions();
}


void Config::create_descriptions()
{

   ConfigBase::create_descriptions();
   
   m_config.add_options()
       ("udp-port",   
               po::value<int>(&m_udp_port)->default_value(m_udp_port_default), 
               "UDP port on which to listen for external messages"
       )
       ("zmq-backend",   
               po::value<std::string>(&m_zmq_backend)->default_value(m_zmq_backend_default), 
               "ZMQ broker backend from which to listen for external messages bearing NPY serialized arrays"
       )
       ("include-path,I", 
               po::value< std::vector<std::string> >()->composing(), 
               "include path"
       )
       ;


   //  default po priority ordering "first setting wins" 
   //  is not convenient for enabling the live override of settings 
   //  where the desired sequence is : config file, command line, live setting 1, live setting 2
   //  ... attempt workaround by using composing and picking the last value 
   //
   m_live.add_options()
       ("yfov",   
               po::value<std::vector<float>>()
                      ->default_value(boost::assign::list_of(0), "0")
                      ->composing()
                      ->notifier(boost::bind(&Config::Yfov_set, this,_1)), 
               "Vertical Field of view in degrees"
       )
       ;

   m_hidden.add_options()
       ("input-file", 
               po::value< std::vector<std::string> >(),  
               "input file"
       )
       ;

   m_cmdline_options.add(m_generic).add(m_config).add(m_hidden).add(m_live);
   m_config_file_options.add(m_config).add(m_hidden).add(m_live);
   m_visible_options.add(m_generic).add(m_config).add(m_live);
   m_positional_options.add("input-file", -1);
}

void Config::Yfov_set(std::vector<float> yfov)
{
   for(auto v:yfov) std::cout << "Config::Yfov_set " << v << '\n' ;
   //std::cout << "Yfov_set " <<  m_yfov << " : size " << yfov.size() << "\n"; 
}


int Config::getUdpPort()
{
    return m_udp_port ;
}
const char* Config::getZMQBackend()
{
    return m_zmq_backend.c_str() ;
}
float Config::getYfov()
{
    return m_yfov ;  
}



void Config::dump()
{
    ConfigBase::dump();

    // hmm need a structured way of doing this, 
    // holding values together with defaults 
    std::cout << "UDP port level is " << m_udp_port << "\n";                
    std::cout << "ZMQ Backend " << m_zmq_backend << "\n";                
    std::cout << "Yfov               " << m_yfov << "\n";                

}

