#pragma once

struct OTimes {

   unsigned int count ; 
   double validate ; 
   double compile ; 
   double prelaunch ; 
   double launch ; 
   const char* _description ; 

   OTimes() :
      count(0),
      validate(0),
      compile(0),
      prelaunch(0),
      launch(0),
      _description(0)
   {
   }

   const char* description(const char* msg="OTimes::description");

};


