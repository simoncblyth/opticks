#pragma once
/**
STimes
=======

struct to hold time measurements


**/
#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API STimes {

   unsigned int count ; 
   double validate ; 
   double compile ; 
   double prelaunch ; 
   double launch ; 
   const char* _description ; 

   STimes() :
      count(0),
      validate(0),
      compile(0),
      prelaunch(0),
      launch(0),
      _description(0)
   {
   }

   const char* description(const char* msg="STimes::description");
   std::string brief(const char* msg="STimes::brief");
   std::string desc();

};


