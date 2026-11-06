#pragma once
/**
srpn.h : 32 bit CSG node representation intended for use with RPN Reverse Polish Notation serialization
==========================================================================================================

All node groups need typecode.

opernode
    typecode[6] leftspan[11] rightspan[11] spare[4] (6+11+11+4 = 32)

listnode
    typecode[6] listprim[10] spare[16] (6+10+16 = 32)

primnode
    typecode[6] + complement[1] + identity[1] + index[16] + spare[8] (6+1+1+16+8 = 32)


As could not fit large enough separate indices for param and transforms
into the 32 bits of primnode - decide to share an index between param
and transforms and add an identity bit to avoid pointless access to
identity transforms. This shared index means that rpn serialization will need
to also populate rpn_param and rpn_transform buffers minting the shared
index as it goes.

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SRPN_METHOD __host__ __device__ __forceinline__
#else
#    define SRPN_METHOD inline
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "OpticksCSG_6bit.h"  // will switch to this as standard in future

#endif


struct srpn
{
    unsigned data;

    SRPN_METHOD srpn() : data(0) {}
    SRPN_METHOD explicit srpn(unsigned raw) : data(raw) {}


    SRPN_METHOD void set_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan );  // 6 + 11 + 11
    SRPN_METHOD int  typecode() const ;
    SRPN_METHOD int  leftspan() const ;
    SRPN_METHOD int  rightspan() const ;

    SRPN_METHOD void set_listnode( unsigned typecode, unsigned num_listprim );   // 6 + 10
    SRPN_METHOD int  listprim() const ;

    SRPN_METHOD void set_primnode( unsigned typecode, bool complement, bool identity, unsigned index ); // 6 + 1 + 1 + 16
    SRPN_METHOD bool identity() const ;
    SRPN_METHOD bool complement() const ;
    SRPN_METHOD int  index() const ;  // shared index into rpn_param + rpn_tran buffers

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ;
#endif


};


SRPN_METHOD void srpn::set_opernode(unsigned typecode, unsigned leftspan, unsigned rightspan) // 6 + 11 + 11
{
    data = (typecode & 0x3Fu) |
           ((leftspan & 0x7FFu) << 6) |
           ((rightspan & 0x7FFu) << 16);
}

SRPN_METHOD int srpn::typecode() const { return  ( data >> 0 ) & 0x3Fu ; }
SRPN_METHOD int srpn::leftspan() const { return  ( data >> 6 ) & 0x7FFu ; }
SRPN_METHOD int srpn::rightspan() const { return ( data >> 16 ) & 0x7FFu ; }


SRPN_METHOD void srpn::set_listnode(unsigned typecode, unsigned num_listprim) // 6 + 10
{
    data = (typecode & 0x3Fu) |
          ((num_listprim & 0x3FFu) << 6);
}
SRPN_METHOD int srpn::listprim() const { return (data >> 6) & 0x3FFu; }


SRPN_METHOD void srpn::set_primnode(unsigned typecode, bool complement, bool identity, unsigned index) // 6 + 1 + 1 + 16
{
    data = (typecode & 0x3Fu) |
           ((complement ? 1u : 0u) << 6) |
           ((identity ? 1u : 0u) << 7) |
           ((index & 0xFFFFu) << 8);
}
SRPN_METHOD bool srpn::identity() const   { return ((data >> 7) & 0x1u) != 0; }
SRPN_METHOD bool srpn::complement() const { return ((data >> 6) & 0x1u) != 0; }
SRPN_METHOD int srpn::index() const {       return (data >> 8) & 0xFFFFu; }


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SRPN_METHOD std::string srpn::desc() const
{
    int tc = typecode();
    std::stringstream ss ;

    if(CSG::IsOperator(tc))       ss << "OPER " ;
    else if(CSG::IsList(tc))      ss << "LIST " ;
    else if(CSG::IsPrimitive(tc)) ss << "PRIM " ;
    else                          ss << "ERR? " ;

    ss << std::setw(25) << CSG::Name(tc) << " " ;

    if(CSG::IsOperator(tc))
    {
        ss << "left      " << std::setw(4) << leftspan()  << " " ;
        ss << "right     " << std::setw(4) << rightspan() << " " ;
    }
    else if(CSG::IsList(tc))
    {
        ss << "listprim  " << std::setw(4) << listprim() << " " ;
    }
    else if(CSG::IsPrimitive(tc))
    {
        ss << "identity    " << std::setw(1) << ( identity() ? "Y" : "N" )   << " " ;
        ss << "complement  " << std::setw(1) << ( complement() ? "Y" : "N" ) << " ";
        ss << "index       " << std::setw(10) << index() << " " ;
    }
    std::string str = ss.str() ;
    return str ;
}

#endif



