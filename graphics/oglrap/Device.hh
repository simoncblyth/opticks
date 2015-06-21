#pragma once

#include <vector>

//class NPYBase ; 

class Device {
    public:
        Device();
        void add(void* smth);
        bool isUploaded(void* smth);

    private:
        std::vector<void*> m_uploads ; 

};


inline Device::Device()
{
}

inline void Device::add(void* smth)
{
    m_uploads.push_back(smth);
}


inline bool Device::isUploaded(void* smth)
{
    for(unsigned int i=0 ; i < m_uploads.size() ; i++) if(m_uploads[i] == smth) return true ;   
    return false ; 
}

