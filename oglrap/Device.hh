#pragma once

#include <vector>

#include "OGLRAP_API_EXPORT.hh"
#include "OGLRAP_HEAD.hh"

class OGLRAP_API Device {
    public:
        Device();
        void add(void* smth);
        bool isUploaded(void* smth);
    private:
        std::vector<void*> m_uploads ; 

};
#include "OGLRAP_TAIL.hh"


