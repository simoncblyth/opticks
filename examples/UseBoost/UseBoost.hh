#pragma once 

#define API  __attribute__ ((visibility ("default")))

struct API UseBoost 
{
   //static const char* program_location(); 
   static const char* concat_path( int argc, char** argv );
   static void dump_file_size(const char* path);


};




