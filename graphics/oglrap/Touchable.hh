#pragma once

class Touchable {
    public:
        virtual ~Touchable(){}
        virtual unsigned int touch(int ix, int iy) = 0 ;
  
};
