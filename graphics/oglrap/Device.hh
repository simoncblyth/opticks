#pragma once

#include <vector>

class NPYBase ; 

class Device {
    public:
        Device();
        void add(NPYBase* npy);
        bool isUploaded(NPYBase* npy);

    private:
        std::vector<NPYBase*> m_uploads ; 

};


inline Device::Device()
{
}

inline void Device::add(NPYBase* npy)
{
    m_uploads.push_back(npy);
}

inline bool Device::isUploaded(NPYBase* npy)
{
    for(unsigned int i=0 ; i < m_uploads.size() ; i++) if(m_uploads[i] == npy) return true ;   
    return false ; 
}

