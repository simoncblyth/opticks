#pragma once

template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksProfile 
{
    public:
       OpticksProfile();
       void stamp(const char* tag);
       void save();
    private:
       float       m_vm0 ; 
       NPY<float>* m_vm ; 

};

#include "OKCORE_TAIL.hh"



