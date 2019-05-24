#pragma once

#include <vector>
#include <string>
#include <array>
#include <map>

class NPYBase ; 
class NPYSpec ; 
class NPYSpecList ; 


#include "NPY_API_EXPORT.hh"

/**
NPYList
=========

**/

class NPY_API NPYList
{
    public:
        enum { MAX_BUF = 16 } ;
    public:
        NPYList(const NPYSpecList* specs);
    private:
        void init();
    public:
        NPYBase::Type_t getBufferType( int bid ) const ;
        const char*     getBufferName( int bid ) const ;
        const NPYSpec*  getBufferSpec( int bid ) const ;
        std::string     getBufferPath( const char* treedir, int bid ) const ;
        std::string     desc() const ; 
    public:
        void            setLocked(int bid, bool locked=true) ; 
        bool            isLocked(int bid) ; 
        NPYBase*        getBuffer(int bid) const  ; 
        std::string     getBufferShape(int bid) const  ;
        unsigned        getNumItems(int bid) const  ;
    public:
        void            saveBuffer(const char* treedir, int bid , const char* msg=NULL ) const ;
        void            initBuffer(int bid, int ni, bool zero, const char* msg=NULL) ; 
        void            setBuffer(int bid, NPYBase* buffer , const char* msg=NULL ) ; 
        void            loadBuffer(const char* treedir, int bid , const char* msg=NULL ) ;
    private:
        const NPYSpecList*               m_specs ; 
        std::array<NPYBase*, MAX_BUF>    m_buf ;
        std::array<bool,     MAX_BUF>    m_locked ;



}; 
 
