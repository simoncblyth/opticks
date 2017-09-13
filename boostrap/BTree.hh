#pragma once

#include <boost/property_tree/ptree.hpp>
#include "BRAP_HEAD.hh"


class BTree {
   public:
      static void saveTree(const boost::property_tree::ptree& t , const char* path);
      static int loadTree(boost::property_tree::ptree& t , const char* path);
      static int loadJSONString(boost::property_tree::ptree& t , const char* json) ; 


};

#include "BRAP_TAIL.hh"



