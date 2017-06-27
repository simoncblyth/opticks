#pragma once

#include <string>
#include "NPY_API_EXPORT.hh"

typedef enum {

  UNCLASSIFIED = 0, 

  XMIN_INSIDE = 0x1 << 0, 
  YMIN_INSIDE = 0x1 << 1, 
  ZMIN_INSIDE = 0x1 << 2, 

  XMAX_INSIDE = 0x1 << 3, 
  YMAX_INSIDE = 0x1 << 4, 
  ZMAX_INSIDE = 0x1 << 5,
 
  XMIN_COINCIDENT = 0x1 << 6, 
  YMIN_COINCIDENT = 0x1 << 7, 
  ZMIN_COINCIDENT = 0x1 << 8, 

  XMAX_COINCIDENT = 0x1 << 9, 
  YMAX_COINCIDENT = 0x1 << 10, 
  ZMAX_COINCIDENT = 0x1 << 11,
  
  XMIN_OUTSIDE = 0x1 << 12, 
  YMIN_OUTSIDE = 0x1 << 13, 
  ZMIN_OUTSIDE = 0x1 << 14, 

  XMAX_OUTSIDE = 0x1 << 15, 
  YMAX_OUTSIDE = 0x1 << 16, 
  ZMAX_OUTSIDE = 0x1 << 17

} NBBoxContainment_t ;


struct NPY_API nbboxenum 
{
    static std::string ContainmentMaskString( unsigned mask ); 
    static const char* ContainmentName( NBBoxContainment_t cont );

    static const char* UNCLASSIFIED_ ; 

    static const char* XMIN_INSIDE_ ; 
    static const char* YMIN_INSIDE_ ; 
    static const char* ZMIN_INSIDE_ ; 
    static const char* XMAX_INSIDE_ ; 
    static const char* YMAX_INSIDE_ ; 
    static const char* ZMAX_INSIDE_ ; 

    static const char* XMIN_COINCIDENT_ ; 
    static const char* YMIN_COINCIDENT_ ; 
    static const char* ZMIN_COINCIDENT_ ; 
    static const char* XMAX_COINCIDENT_ ; 
    static const char* YMAX_COINCIDENT_ ; 
    static const char* ZMAX_COINCIDENT_ ; 

    static const char* XMIN_OUTSIDE_ ; 
    static const char* YMIN_OUTSIDE_ ; 
    static const char* ZMIN_OUTSIDE_ ; 
    static const char* XMAX_OUTSIDE_ ; 
    static const char* YMAX_OUTSIDE_ ; 
    static const char* ZMAX_OUTSIDE_ ; 

};


