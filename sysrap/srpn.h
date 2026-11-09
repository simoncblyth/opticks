#pragma once
/**
srpn.h : 32 bit CSG node representation intended for use with RPN Reverse Polish Notation serialization
==========================================================================================================

All node groups need typecode.

opernode
    typecode[7] leftspan[11] rightspan[11] spare[3] (7+11+11+3 = 32)

listnode
    typecode[7] listprim[10] spare[15] (7+10+15 = 32)

primnode
    typecode[7] + complement[1] + identity[1] + index[16] + spare[7] (7+1+1+16+7 = 32)


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
#include "OpticksCSG.h"  // will switch to this as standard in future

#endif


struct srpn
{
    unsigned data;

    SRPN_METHOD srpn() : data(0) {}
    SRPN_METHOD explicit srpn(unsigned raw) : data(raw) {}


    SRPN_METHOD void set_opernode( unsigned typecode, unsigned leftspan, unsigned rightspan );  // 7 + 11 + 11
    SRPN_METHOD int  typecode() const ;
    SRPN_METHOD int  leftspan() const ;
    SRPN_METHOD int  rightspan() const ;

    SRPN_METHOD void set_listnode( unsigned typecode, unsigned num_listprim );   // 7 + 10
    SRPN_METHOD int  listprim() const ;

    SRPN_METHOD void set_primnode( unsigned typecode, bool complement, bool identity, unsigned index ); // 7 + 1 + 1 + 16
    SRPN_METHOD bool identity() const ;
    SRPN_METHOD bool complement() const ;
    SRPN_METHOD int  index() const ;  // shared index into rpn_param + rpn_tran buffers

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ;
#endif


};


SRPN_METHOD void srpn::set_opernode(unsigned typecode, unsigned leftspan, unsigned rightspan) // 7 + 11 + 11
{
    data = (typecode & 0x7Fu) |
           ((leftspan & 0x7FFu) << 7) |
           ((rightspan & 0x7FFu) << 17);
}

SRPN_METHOD int srpn::typecode() const { return  ( data >> 0 ) & 0x7Fu ; }
SRPN_METHOD int srpn::leftspan() const { return  ( data >> 7 ) & 0x7FFu ; }
SRPN_METHOD int srpn::rightspan() const { return ( data >> 17 ) & 0x7FFu ; }


SRPN_METHOD void srpn::set_listnode(unsigned typecode, unsigned num_listprim) // 7 + 10
{
    data = (typecode & 0x7Fu) |
          ((num_listprim & 0x3FFu) << 7);
}
SRPN_METHOD int srpn::listprim() const { return (data >> 7) & 0x3FFu; }


SRPN_METHOD void srpn::set_primnode(unsigned typecode, bool complement, bool identity, unsigned index) // 7 + 1 + 1 + 16
{
    data = (typecode & 0x7Fu) |
           ((complement ? 1u : 0u) << 7) |
           ((identity ? 1u : 0u) << 8) |
           ((index & 0xFFFFu) << 9);
}
SRPN_METHOD bool srpn::complement() const { return ((data >> 7) & 0x1u) != 0; }
SRPN_METHOD bool srpn::identity() const   { return ((data >> 8) & 0x1u) != 0; }
SRPN_METHOD int srpn::index() const {       return (data >> 9) & 0xFFFFu; }


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



