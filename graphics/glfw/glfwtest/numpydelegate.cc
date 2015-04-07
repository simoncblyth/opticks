#include "numpydelegate.hh"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <boost/thread.hpp>

#include "numpyserver.hpp"
#include "Cfg.hh"


numpydelegate::numpydelegate()
     :
     m_server(NULL),
     m_udp_port(8080),
     m_npy_echo(0),
     m_zmq_backend("tcp://127.0.0.1:5002")
{
}

void numpydelegate::setServer(numpyserver<numpydelegate>* server)
{
    m_server = server ; 
}  

void numpydelegate::configureI(const char* name, std::vector<int> values)
{
    if(values.empty()) return ;
    if(strcmp(name, "udpport")==0) setUDPPort(values.back());
    if(strcmp(name, "npyecho")==0) setNPYEcho(values.back());
}
void numpydelegate::configureS(const char* name, std::vector<std::string> values)
{
    if(values.empty()) return ;
    if(strcmp(name, "zmqbackend")==0) setZMQBackend(values.back());
}

void numpydelegate::setUDPPort(int port)
{
    m_udp_port = port ;
}
void numpydelegate::setNPYEcho(int echo)
{
    m_npy_echo = echo ;
}
void numpydelegate::setZMQBackend(std::string& backend)
{
    m_zmq_backend = backend ;
}

int numpydelegate::getUDPPort()
{
    return m_udp_port ;    
}
int numpydelegate::getNPYEcho()
{
    return m_npy_echo ;    
}
std::string& numpydelegate::getZMQBackend()
{
    return m_zmq_backend ;    
}


void numpydelegate::addLiveCfg(Cfg* cfg)
{
    m_live_cfg.push_back(cfg);
}

void numpydelegate::interpretExternalMessage(std::string msg)
{
    for(size_t i=0 ; i<m_live_cfg.size() ; ++i)
    {
        Cfg* cfg = m_live_cfg[i];
        cfg->liveline(msg.c_str());  // cfg listener objects may get live configured 
    }
}

void numpydelegate::on_msg(std::string _addr, unsigned short port, std::string msg)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " numpydelegate::on_msg " 
              << " addr ["  << _addr << "] "
              << " port ["  << port << "] "
              << " msg ["  << msg << "] "
              << std::endl;


    interpretExternalMessage(msg);

    std::stringstream ss ; 
    ss <<  msg ;
    ss << " returned from numpydelegate " ;

    boost::shared_ptr<std::string> addr(new std::string(_addr));
    boost::shared_ptr<std::string> reply(new std::string(ss.str()));

    m_server->send( *addr, port, *reply );  
    // is passing a deref-ed shared_ptr simply a cunning way to leak  
}

void numpydelegate::on_npy(std::vector<int> _shape, std::vector<float> _data, std::string _metadata)
{
    assert(_shape.size() == 3); 
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " numpydelegate::on_npy "
              << " shape dimension " << _shape.size()
              << _shape[0] << "/"
              << _shape[1] << "/"
              << _shape[2] << " "
              << " data size " << _data.size()
              << " metadata [" << _metadata << "]" 
              << std::endl ; 

  // copying and replying

   boost::shared_ptr<std::vector<int>>   shape(new std::vector<int>(_shape)); 
   boost::shared_ptr<std::vector<float>> data(new std::vector<float>(_data)); 
   boost::shared_ptr<std::string>        metadata(new std::string(_metadata));

   //for(size_t i=0 ; i< data->size() ; ++i) (*data)[i] = (*data)[i]*2.0f ; 

   m_server->response( *shape, *data, *metadata );  

} 




