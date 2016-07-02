#pragma once

#include <vector>
class ViewNPY ; 

#include "NPY_API_EXPORT.hh"

#ifdef _MSC_VER
#pragma warning(push)
// members needs to have dll-interface to be used by clients
#pragma warning( disable : 4251 )
#endif


class NPY_API MultiViewNPY {
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


#ifdef _MSC_VER
#pragma warning(pop)
#endif


