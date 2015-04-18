#pragma once

#include <vector>
class VecNPY ; 

class MultiVecNPY {
    public:
        MultiVecNPY();

    public:
        void add(VecNPY* vec);
        VecNPY* operator [](const char* name);
        VecNPY* operator [](unsigned int index);
        unsigned int getNumVecs();


        void Summary(const char* msg="MultiVecNPY::Summary");
        void Print(const char* msg="MultiVecNPY::Print");

    private:
        VecNPY* find(const char* name);

    private:
        std::vector<VecNPY*> m_vecs ;  

};


