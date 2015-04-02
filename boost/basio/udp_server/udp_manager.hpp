
#pragma once


#include <string>
#include <iomanip>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include "udp_server.hpp"


// following /opt/local/share/doc/boost/doc/html/boost_asio/example/cpp03/services/logger_service.hpp

//
// udp_manager class is a layer inbetween the application 
// and the udp_server that controls the threading 
// so the application doesnt need to worry about it 
//
// instanciating the manager 
//
//    * kicks off worker thread and work_io_service_ together 
//      with dummy work to keep it alive and the udp_server
//




class udp_manager {

    boost::asio::io_service&                         m_main_io_service;
    boost::asio::io_service                          m_local_io_service;
    boost::scoped_ptr<boost::asio::io_service::work> m_work ;
    boost::scoped_ptr<boost::thread>                 m_work_thread;
    udp_server                                       m_udp_server ; 


public:

   udp_manager(boost::asio::io_service& io_service, unsigned int port)
       : 
       m_main_io_service(io_service),
       m_local_io_service(),
       m_work(new boost::asio::io_service::work(m_local_io_service)), // keeps m_work_thread/m_local_io_service::run from terminating 
       m_work_thread(new boost::thread(boost::bind(
                                &boost::asio::io_service::run, 
                                &m_local_io_service
                                   ))),
       m_udp_server(m_local_io_service, m_main_io_service, this, port)   
    {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " udp_manager::udp_manager " << std::endl;
    }

    boost::asio::io_service& get_io_service()
    {
        return m_main_io_service ; 
    }

    // hmm better to pass a delegate object to receive the message
    void on_message_0(boost::shared_ptr<std::string> msg)
    {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " udp_manager::on_message_0 " 
                  << " msg "  << *msg 
                  << std::endl;
    }

    void on_message(std::string msg)
    {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " udp_manager::on_message " 
                  << " msg ["  << msg << "] "
                  << std::endl;
    }




};



