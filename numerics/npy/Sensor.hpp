#pragma once


class Sensor {
   public:  
       Sensor();
       void load(const char* idpath, const char* ext="idmap");
   private:
       void read(const char* path);
        
};


inline Sensor::Sensor()
{
}


