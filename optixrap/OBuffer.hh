#pragma once

/**
OBuffer
==========

**/


#include "OXRAP_API_EXPORT.hh"
#include <string>

class OXRAP_API OBuffer {
public:

  static const char* RT_BUFFER_INPUT_ ; 
  static const char* RT_BUFFER_OUTPUT_ ; 
  static const char* RT_BUFFER_INPUT_OUTPUT_ ; 
  static const char* RT_BUFFER_PROGRESSIVE_STREAM_ ; 

  static const char* RT_BUFFER_LAYERED_ ; 
  static const char* RT_BUFFER_CUBEMAP_ ; 
  static const char* RT_BUFFER_GPU_LOCAL_ ; 
  static const char* RT_BUFFER_COPY_ON_DIRTY_ ; 

  static std::string BufferDesc(unsigned buffer_desc); 
  static void Dump(const char* msg="OBuffer::Dump");

};
 


