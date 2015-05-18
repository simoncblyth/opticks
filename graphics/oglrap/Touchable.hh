#pragma once

class Touchable {
    public:
        virtual ~Touchable(){}
        virtual void touch(unsigned char key, int ix, int iy) = 0 ;
  
};
