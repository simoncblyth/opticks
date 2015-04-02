
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

template <class Delegate>
class udp_manager {

    boost::asio::io_service                          m_local_io_service;
    boost::scoped_ptr<boost::asio::io_service::work> m_local_io_service_work ;
    boost::scoped_ptr<boost::thread>                 m_work_thread;
    udp_server<Delegate>                             m_udp_server ; 

public:

   udp_manager(Delegate* delegate, unsigned int port)
       : 
       m_local_io_service(),
       m_local_io_service_work(new boost::asio::io_service::work(m_local_io_service)), 
       m_work_thread(new boost::thread(boost::bind(
                                &boost::asio::io_service::run, 
                                &m_local_io_service
                                   ))),
       m_udp_server(m_local_io_service, delegate, port)   
    {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " udp_manager::udp_manager " << std::endl;
    }

};



