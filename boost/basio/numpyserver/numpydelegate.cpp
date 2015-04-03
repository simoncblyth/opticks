#include "numpydelegate.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <boost/thread.hpp>

#include "numpyserver.hpp"

numpydelegate::numpydelegate()
     :
     m_server(NULL)
{
}

void numpydelegate::setServer(numpyserver<numpydelegate>* server)
{
    m_server = server ; 
}  


void numpydelegate::on_msg(std::string _addr, unsigned short port, std::string msg)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " numpydelegate::on_msg " 
              << " addr ["  << _addr << "] "
              << " port ["  << port << "] "
              << " msg ["  << msg << "] "
              << std::endl;


    std::stringstream ss ; 
    ss <<  msg ;
    ss << " returned from numpydelegate " ;

    boost::shared_ptr<std::string> addr(new std::string(_addr));
    boost::shared_ptr<std::string> reply(new std::string(ss.str()));

    m_server->send( *addr, port, *reply );  
    // is passing a deref-ed shared_ptr simply a cunning way to leak  
}

void numpydelegate::on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " numpydelegate::on_npy "
              << " shape dimension " << shape.size()
              << " data size " << data.size()
              << " metadata [" << metadata << "]" 
              << std::endl ; 
} 


