#include "numpydelegate.hh"

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




