/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "PLOG.hh"
#include "NPYBase.hpp"
#include "OFormat.hh"


unsigned long long OFormat::ElementSizeInBytes(RTformat format) // static 
{
    //  OptiX_600:rtuGetSizeForRTformat gives randoms for RT_FORMAT_USER
    //  so return zero 
    size_t element_size ; 

    if( format == RT_FORMAT_USER ) 
    { 
        element_size = 0 ; 
    } 
    else
    {
        rtuGetSizeForRTformat( format, &element_size);
    }


    unsigned long long esz = element_size ; 
    return esz ; 
}


unsigned long long OFormat::Multiplicity(RTformat format) // static
{
   unsigned mul(0) ;
   switch(format)
   {
      case RT_FORMAT_UNKNOWN: mul=0 ; break ; 

      case RT_FORMAT_FLOAT:   mul=1 ; break ;
      case RT_FORMAT_FLOAT2:  mul=2 ; break ;
      case RT_FORMAT_FLOAT3:  mul=3 ; break ;
      case RT_FORMAT_FLOAT4:  mul=4 ; break ;

      case RT_FORMAT_BYTE:    mul=1 ; break ;
      case RT_FORMAT_BYTE2:   mul=2 ; break ;
      case RT_FORMAT_BYTE3:   mul=3 ; break ;
      case RT_FORMAT_BYTE4:   mul=4 ; break ;

      case RT_FORMAT_UNSIGNED_BYTE:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE4: mul=4 ; break ;

      case RT_FORMAT_SHORT:  mul=1 ; break ;
      case RT_FORMAT_SHORT2: mul=2 ; break ;
      case RT_FORMAT_SHORT3: mul=3 ; break ;
      case RT_FORMAT_SHORT4: mul=4 ; break ;

#if OPTIX_VERSION > 3080
      case RT_FORMAT_HALF:  mul=1 ; break ;
      case RT_FORMAT_HALF2: mul=2 ; break ;
      case RT_FORMAT_HALF3: mul=3 ; break ;
      case RT_FORMAT_HALF4: mul=4 ; break ;
#endif

      case RT_FORMAT_UNSIGNED_SHORT:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT4: mul=4 ; break ;

      case RT_FORMAT_INT:  mul=1 ; break ;
      case RT_FORMAT_INT2: mul=2 ; break ;
      case RT_FORMAT_INT3: mul=3 ; break ;
      case RT_FORMAT_INT4: mul=4 ; break ;

      case RT_FORMAT_UNSIGNED_INT:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_INT2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_INT3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_INT4: mul=4 ; break ;

      case RT_FORMAT_USER:       mul=0 ; break ;
      case RT_FORMAT_BUFFER_ID:  mul=0 ; break ;
      case RT_FORMAT_PROGRAM_ID: mul=0 ; break ; 

#if OPTIX_VERSION >= 60000
       case RT_FORMAT_LONG_LONG:   mul=1 ; break ; 
       case RT_FORMAT_LONG_LONG2:  mul=2 ; break ; 
       case RT_FORMAT_LONG_LONG3:  mul=3 ; break ; 
       case RT_FORMAT_LONG_LONG4:  mul=4 ; break ; 
   
       case RT_FORMAT_UNSIGNED_LONG_LONG:   mul=1 ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG2:  mul=2 ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG3:  mul=3 ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG4:  mul=4 ; break ; 

       case RT_FORMAT_UNSIGNED_BC1:  mul=1 ; break ; 
       case RT_FORMAT_UNSIGNED_BC2:  mul=2 ; break ; 
       case RT_FORMAT_UNSIGNED_BC3:  mul=3 ; break ; 
       case RT_FORMAT_UNSIGNED_BC4:  mul=4 ; break ; 
       case RT_FORMAT_UNSIGNED_BC5:  mul=5 ; break ; 
       case RT_FORMAT_UNSIGNED_BC6H:  mul=6 ; break ; 
       case RT_FORMAT_UNSIGNED_BC7:   mul=7 ; break ; 

       case RT_FORMAT_BC4:  mul=4 ; break ; 
       case RT_FORMAT_BC5:  mul=5 ; break ; 
       case RT_FORMAT_BC6H: mul=6 ; break ; 
#endif
   }

   unsigned long long lm = mul ;  
   return lm ; 
}



const char* OFormat::FormatName(RTformat format) // static 
{
   const char* name = NULL ; 
   switch(format)
   {
      case RT_FORMAT_UNKNOWN: name=_RT_FORMAT_UNKNOWN ; break ; 

      case RT_FORMAT_FLOAT:   name=_RT_FORMAT_FLOAT ; break ;
      case RT_FORMAT_FLOAT2:  name=_RT_FORMAT_FLOAT2 ; break ;
      case RT_FORMAT_FLOAT3:  name=_RT_FORMAT_FLOAT3 ; break ;
      case RT_FORMAT_FLOAT4:  name=_RT_FORMAT_FLOAT4 ; break ;

      case RT_FORMAT_BYTE:    name=_RT_FORMAT_BYTE ; break ;
      case RT_FORMAT_BYTE2:   name=_RT_FORMAT_BYTE2 ; break ;
      case RT_FORMAT_BYTE3:   name=_RT_FORMAT_BYTE3 ; break ;
      case RT_FORMAT_BYTE4:   name=_RT_FORMAT_BYTE4 ; break ;

      case RT_FORMAT_UNSIGNED_BYTE:  name=_RT_FORMAT_UNSIGNED_BYTE ; break ;
      case RT_FORMAT_UNSIGNED_BYTE2: name=_RT_FORMAT_UNSIGNED_BYTE2 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE3: name=_RT_FORMAT_UNSIGNED_BYTE3 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE4: name=_RT_FORMAT_UNSIGNED_BYTE4 ; break ;

      case RT_FORMAT_SHORT:  name=_RT_FORMAT_SHORT ; break ;
      case RT_FORMAT_SHORT2: name=_RT_FORMAT_SHORT2 ; break ;
      case RT_FORMAT_SHORT3: name=_RT_FORMAT_SHORT3 ; break ;
      case RT_FORMAT_SHORT4: name=_RT_FORMAT_SHORT4 ; break ;

#if OPTIX_VERSION > 3080
      case RT_FORMAT_HALF:  name=_RT_FORMAT_HALF ; break ;
      case RT_FORMAT_HALF2: name=_RT_FORMAT_HALF2 ; break ;
      case RT_FORMAT_HALF3: name=_RT_FORMAT_HALF3 ; break ;
      case RT_FORMAT_HALF4: name=_RT_FORMAT_HALF4 ; break ;
#endif

      case RT_FORMAT_UNSIGNED_SHORT:  name=_RT_FORMAT_UNSIGNED_SHORT ; break ;
      case RT_FORMAT_UNSIGNED_SHORT2: name=_RT_FORMAT_UNSIGNED_SHORT2 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT3: name=_RT_FORMAT_UNSIGNED_SHORT3 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT4: name=_RT_FORMAT_UNSIGNED_SHORT4 ; break ;

      case RT_FORMAT_INT:  name=_RT_FORMAT_INT ; break ;
      case RT_FORMAT_INT2: name=_RT_FORMAT_INT2 ; break ;
      case RT_FORMAT_INT3: name=_RT_FORMAT_INT3 ; break ;
      case RT_FORMAT_INT4: name=_RT_FORMAT_INT4 ; break ;

      case RT_FORMAT_UNSIGNED_INT:  name=_RT_FORMAT_UNSIGNED_INT ; break ;
      case RT_FORMAT_UNSIGNED_INT2: name=_RT_FORMAT_UNSIGNED_INT2 ; break ;
      case RT_FORMAT_UNSIGNED_INT3: name=_RT_FORMAT_UNSIGNED_INT3 ; break ;
      case RT_FORMAT_UNSIGNED_INT4: name=_RT_FORMAT_UNSIGNED_INT4 ; break ;

      case RT_FORMAT_USER:       name=_RT_FORMAT_USER ; break ;
      case RT_FORMAT_BUFFER_ID:  name=_RT_FORMAT_BUFFER_ID ; break ;
      case RT_FORMAT_PROGRAM_ID: name=_RT_FORMAT_PROGRAM_ID ; break ; 

#if OPTIX_VERSION >= 60000
       case RT_FORMAT_LONG_LONG:    name=_RT_FORMAT_LONG_LONG ; break ; 
       case RT_FORMAT_LONG_LONG2:   name=_RT_FORMAT_LONG_LONG2 ; break ; 
       case RT_FORMAT_LONG_LONG3:   name=_RT_FORMAT_LONG_LONG3 ; break ; 
       case RT_FORMAT_LONG_LONG4:   name=_RT_FORMAT_LONG_LONG4 ; break ; 

       case RT_FORMAT_UNSIGNED_LONG_LONG:    name=_RT_FORMAT_UNSIGNED_LONG_LONG ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG2:   name=_RT_FORMAT_UNSIGNED_LONG_LONG2 ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG3:   name=_RT_FORMAT_UNSIGNED_LONG_LONG3 ; break ; 
       case RT_FORMAT_UNSIGNED_LONG_LONG4:   name=_RT_FORMAT_UNSIGNED_LONG_LONG4 ; break ; 
   
       case RT_FORMAT_UNSIGNED_BC1:  name = _RT_FORMAT_UNSIGNED_BC1 ; break ; 
       case RT_FORMAT_UNSIGNED_BC2:  name = _RT_FORMAT_UNSIGNED_BC2 ; break ; 
       case RT_FORMAT_UNSIGNED_BC3:  name = _RT_FORMAT_UNSIGNED_BC3 ; break ; 
       case RT_FORMAT_UNSIGNED_BC4:  name = _RT_FORMAT_UNSIGNED_BC4 ; break ; 
       case RT_FORMAT_UNSIGNED_BC5:  name = _RT_FORMAT_UNSIGNED_BC5 ; break ; 
       case RT_FORMAT_UNSIGNED_BC6H:  name = _RT_FORMAT_UNSIGNED_BC6H ; break ; 
       case RT_FORMAT_UNSIGNED_BC7:  name = _RT_FORMAT_UNSIGNED_BC7 ; break ; 

       case RT_FORMAT_BC4:  name = _RT_FORMAT_BC4 ; break ; 
       case RT_FORMAT_BC5:  name = _RT_FORMAT_BC5 ; break ; 
       case RT_FORMAT_BC6H: name = _RT_FORMAT_BC6H ; break ; 
#endif
   }
   return name ; 
}




   const char* OFormat::_RT_FORMAT_UNKNOWN = "UNKNOWN" ;

   const char* OFormat::_RT_FORMAT_FLOAT = "FLOAT" ;
   const char* OFormat::_RT_FORMAT_FLOAT2 = "FLOAT2" ;
   const char* OFormat::_RT_FORMAT_FLOAT3 = "FLOAT3" ;
   const char* OFormat::_RT_FORMAT_FLOAT4 = "FLOAT4" ;

   const char* OFormat::_RT_FORMAT_BYTE = "BYTE" ;
   const char* OFormat::_RT_FORMAT_BYTE2 = "BYTE2" ;
   const char* OFormat::_RT_FORMAT_BYTE3 = "BYTE3" ;
   const char* OFormat::_RT_FORMAT_BYTE4 = "BYTE4" ;

   const char* OFormat::_RT_FORMAT_UNSIGNED_BYTE = "UNSIGNED_BYTE" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_BYTE2 = "UNSIGNED_BYTE2" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_BYTE3 = "UNSIGNED_BYTE3" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_BYTE4 = "UNSIGNED_BYTE4" ;

   const char* OFormat::_RT_FORMAT_SHORT = "SHORT" ;
   const char* OFormat::_RT_FORMAT_SHORT2 = "SHORT2" ;
   const char* OFormat::_RT_FORMAT_SHORT3 = "SHORT3" ;
   const char* OFormat::_RT_FORMAT_SHORT4 = "SHORT4" ;

#if OPTIX_VERSION > 3080
   const char* OFormat::_RT_FORMAT_HALF = "HALF" ;
   const char* OFormat::_RT_FORMAT_HALF2 = "HALF2" ;
   const char* OFormat::_RT_FORMAT_HALF3 = "HALF3" ;
   const char* OFormat::_RT_FORMAT_HALF4 = "HALF4" ;
#endif

   const char* OFormat::_RT_FORMAT_UNSIGNED_SHORT = "UNSIGNED_SHORT" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_SHORT2 = "UNSIGNED_SHORT2" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_SHORT3 = "UNSIGNED_SHORT3";
   const char* OFormat::_RT_FORMAT_UNSIGNED_SHORT4 = "UNSIGNED_SHORT4";

   const char* OFormat::_RT_FORMAT_INT = "INT" ;
   const char* OFormat::_RT_FORMAT_INT2 = "INT2";
   const char* OFormat::_RT_FORMAT_INT3 = "INT3";
   const char* OFormat::_RT_FORMAT_INT4 = "INT4";

   const char* OFormat::_RT_FORMAT_UNSIGNED_INT = "UNSIGNED_INT" ;
   const char* OFormat::_RT_FORMAT_UNSIGNED_INT2 = "UNSIGNED_INT2";
   const char* OFormat::_RT_FORMAT_UNSIGNED_INT3 = "UNSIGNED_INT3";
   const char* OFormat::_RT_FORMAT_UNSIGNED_INT4 = "UNSIGNED_INT4";

   const char* OFormat::_RT_FORMAT_USER = "USER" ;
   const char* OFormat::_RT_FORMAT_BUFFER_ID = "BUFFER_ID" ;
   const char* OFormat::_RT_FORMAT_PROGRAM_ID = "PROGRAM_ID" ;

#if OPTIX_VERSION >= 60000
    const char* OFormat::_RT_FORMAT_LONG_LONG  = "LONG_LONG" ;  
    const char* OFormat::_RT_FORMAT_LONG_LONG2 = "LONG_LONG2" ; 
    const char* OFormat::_RT_FORMAT_LONG_LONG3 = "LONG_LONG3" ; 
    const char* OFormat::_RT_FORMAT_LONG_LONG4 = "LONG_LONG4" ; 

    const char* OFormat::_RT_FORMAT_UNSIGNED_LONG_LONG  = "UNSIGNED_LONG_LONG" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_LONG_LONG2 = "UNSIGNED_LONG_LONG2" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_LONG_LONG3 = "UNSIGNED_LONG_LONG3" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_LONG_LONG4 = "UNSIGNED_LONG_LONG4" ; 
   
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC1  = "UNSIGNED_BC1" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC2  = "UNSIGNED_BC2" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC3  = "UNSIGNED_BC3" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC4  = "UNSIGNED_BC4" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC5  = "UNSIGNED_BC5" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC6H  = "UNSIGNED_BC6H" ; 
    const char* OFormat::_RT_FORMAT_UNSIGNED_BC7  = "UNSIGNED_BC7" ; 

    const char* OFormat::_RT_FORMAT_BC4  = "BC4" ; 
    const char* OFormat::_RT_FORMAT_BC5  = "BC5" ; 
    const char* OFormat::_RT_FORMAT_BC6H = "BC6H" ; 
#endif










// template specializations for returning the appropriate RTformat for each multiplicity 


template<>
RTformat OFormat::Get<float>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_FLOAT  ; break ; 
       case 2: format=RT_FORMAT_FLOAT2 ; break ; 
       case 3: format=RT_FORMAT_FLOAT3 ; break ; 
       case 4: format=RT_FORMAT_FLOAT4 ; break ; 
    }   
    return format ; 
}

template<>
RTformat OFormat::Get<unsigned char>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_UNSIGNED_BYTE  ; break ; 
       case 2: format=RT_FORMAT_UNSIGNED_BYTE2 ; break ; 
       case 3: format=RT_FORMAT_UNSIGNED_BYTE3 ; break ; 
       case 4: format=RT_FORMAT_UNSIGNED_BYTE4 ; break ; 
    }   
    return format ; 
}

template<>
RTformat OFormat::Get<char>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_BYTE  ; break ; 
       case 2: format=RT_FORMAT_BYTE2 ; break ; 
       case 3: format=RT_FORMAT_BYTE3 ; break ; 
       case 4: format=RT_FORMAT_BYTE4 ; break ; 
    }   
    return format ; 
}

template<>
RTformat OFormat::Get<unsigned>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_UNSIGNED_INT  ; break ; 
       case 2: format=RT_FORMAT_UNSIGNED_INT2 ; break ; 
       case 3: format=RT_FORMAT_UNSIGNED_INT3 ; break ; 
       case 4: format=RT_FORMAT_UNSIGNED_INT4 ; break ; 
    }   
    return format ; 
}


template<>
RTformat OFormat::Get<short>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_SHORT  ; break ; 
       case 2: format=RT_FORMAT_SHORT2 ; break ; 
       case 3: format=RT_FORMAT_SHORT3 ; break ; 
       case 4: format=RT_FORMAT_SHORT4 ; break ; 
    }   
    return format ; 
}

template<>
RTformat OFormat::Get<int>(unsigned multiplicity)
{
    assert( multiplicity > 0 && multiplicity < 5 ); 
    RTformat format ; 
    switch(multiplicity)
    {   
       case 1: format=RT_FORMAT_INT  ; break ; 
       case 2: format=RT_FORMAT_INT2 ; break ; 
       case 3: format=RT_FORMAT_INT3 ; break ; 
       case 4: format=RT_FORMAT_INT4 ; break ; 
    }   
    return format ; 
}


/**
OFormat::ArrayType
--------------------

**/

RTformat OFormat::ArrayType(const NPYBase* arr)
{
   unsigned multiplicity = arr->getShape(-1) ; 
   assert( multiplicity < 5 );  
   NPYBase::Type_t type = arr->getType() ; 

   RTformat format ; 
   switch(type)
   {
       case NPYBase::FLOAT:     format = OFormat::Get<float>(multiplicity)         ; break ; 
       case NPYBase::INT:       format = OFormat::Get<int>(multiplicity)           ; break ; 
       case NPYBase::UINT:      format = OFormat::Get<unsigned>(multiplicity)      ; break ; 
       case NPYBase::SHORT:     format = OFormat::Get<short>(multiplicity)         ; break ;
       case NPYBase::UCHAR:     format = OFormat::Get<unsigned char>(multiplicity) ; break ; 
       case NPYBase::CHAR:      format = OFormat::Get<char>(multiplicity)          ; break ; 
       case NPYBase::DOUBLE:    format = RT_FORMAT_USER                            ; break ;   
       case NPYBase::ULONGLONG: format = RT_FORMAT_USER                            ; break ; 
   }
   return format ; 
}




