#pragma once

#include "net_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

template <class Delegate>
class numpyserver {

   boost::asio::io_service                              m_io_service ;
   boost::scoped_ptr<boost::asio::io_service::work>     m_io_service_work ;
   net_manager<Delegate>                                m_net_manager ; 

public:
   numpyserver(Delegate* delegate)  
      :
        m_io_service_work(new boost::asio::io_service::work(m_io_service)),
        m_net_manager(delegate, m_io_service)
   {

        delegate->setServer(this);  // allows the delegate to reply 

#ifdef VERBOSE
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " numpyserver::numpyserver " << std::endl;

#endif
   }

public:
   void sleep(unsigned int seconds);
   void poll_one();
   void poll();
   void stop();
   boost::asio::io_service& get_io_service();

public:
   // used by delegates that receive on_msg and on_npy posts
   void send(std::string& addr, unsigned short port, std::string& msg );                   // "send" as UDP is connectionless
   void response(std::vector<int>& shape, std::vector<float>& data, std::string& metadata );  // "response" as only works in response to ZMQ REQ

};



template <typename Delegate>
void numpyserver<Delegate>::sleep(unsigned int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

template <typename Delegate>
void numpyserver<Delegate>::poll_one()
{
    //std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::poll_one " << std::endl;
    m_io_service.poll_one();
} 

template <typename Delegate>
void numpyserver<Delegate>::poll()
{
    //std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::poll " << std::endl;
    m_io_service.poll();
} 

template <typename Delegate>
void numpyserver<Delegate>::stop()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " numpyserver::stop " << std::endl;
    m_io_service_work.reset();
} 

template <typename Delegate>
boost::asio::io_service& numpyserver<Delegate>::get_io_service()
{
    return m_io_service ;  
}

template <typename Delegate>
void numpyserver<Delegate>::send(std::string& addr, unsigned short port, std::string& msg )
{
    m_net_manager.send(addr, port, msg);
}


template <typename Delegate>
void numpyserver<Delegate>::response(std::vector<int>& shape,std::vector<float>& data,  std::string& metadata )
{
    m_net_manager.response(shape, data, metadata);
}







