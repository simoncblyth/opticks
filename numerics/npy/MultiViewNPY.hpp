#pragma once

#include <vector>
class ViewNPY ; 

class MultiViewNPY {
    public:
        MultiViewNPY();

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
        std::vector<ViewNPY*> m_vecs ;  

};


