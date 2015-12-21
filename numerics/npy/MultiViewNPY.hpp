#pragma once

#include <vector>
#include <cstring>

class ViewNPY ; 

class MultiViewNPY {
    public:
        MultiViewNPY(const char* name="no-name");
        const char* getName();
    public:
        void add(ViewNPY* vec);
        ViewNPY* operator [](const char* name);
        ViewNPY* operator [](unsigned int index);
        unsigned int getNumVecs();


        void Summary(const char* msg="MultiViewNPY::Summary");
        void Print(const char* msg="MultiViewNPY::Print");

    private:
        ViewNPY* find(const char* name);

    private:
        const char*           m_name ; 
        std::vector<ViewNPY*> m_vecs ;  

};

inline MultiViewNPY::MultiViewNPY(const char* name)
   :   
    m_name(strdup(name))
{
}

inline const char* MultiViewNPY::getName()
{
    return m_name ;
}

