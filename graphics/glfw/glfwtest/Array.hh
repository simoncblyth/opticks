#ifndef ARRAY_H
#define ARRAY_H

class Array {
  public:
     Array(unsigned int length, const float* values);
     virtual ~Array();

     unsigned int getLength();
     const float* getValues();

  private:
     unsigned int m_length ;
     const float* m_values ;

};

#endif

