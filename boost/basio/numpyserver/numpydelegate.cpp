#include "numpydelegate.hpp"

#include <iostream>
#include <iomanip>
#include <boost/thread.hpp>

numpydelegate::numpydelegate()
{
}

void numpydelegate::on_msg(std::string msg)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " numpydelegate::on_msg " 
              << " msg ["  << msg << "] "
              << std::endl;
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


