#pragma once

#include <vector>
#include <string>
#include <set>
#include <map>
#include <cstring>
#include <cstdlib>
#include <cassert>


#include "NPY_API_EXPORT.hh"
#include "NPY_FLAGS.hh"


#include "uif.h"
#include "ucharfour.h"
#include "charfour.h"
#include "numpy.hpp"


#include "NPYBase.hpp"
#include "NQuad.hpp"

struct BBufSpec ; 
struct NSlice ; 
class NPYSpec ; 
class G4StepNPY ; 

/*
Interop NumPy -> NPY
======================

The type of the NumPy array saved from python
needs to match the NPY basis type 
eg NPY<float> NPY<short> NPY<double>

From python control the type, eg to save as float with::

    In [3]: a.dtype
    Out[3]: dtype('float64')

    In [4]: b = np.array(a, dtype=np.float32)

    np.save("/tmp/slowcomponent.npy", b ) 


Going the other way, NPY -> NumPy just works as NumPy 
recognises the type on loading, thanks to the numpy.hpp metadata header.

*/

// hmm most of the inlines should go into .cpp 
// and the NPY_FLAGS should go there too
// otherwise type conversion warning quellings will spread to clients ?


/**

NPY : C++ NumPy style array 
==============================

*NPY* provides array operations includes persistency, the structure 
and approach are inspired by the NumPy array http://www.numpy.org 
and the **NPY** persistency format is adopted allowing Opticks
geometry and event data to be loaded into python OR ipython sessions
with::

   import numpy as np
   a = np.load("/path/to/file.npy")

**/

template <class T>
class NPY_API NPY : public NPYBase {

   friend class SeqNPY ; 
   friend class AxisNPY ; 
   friend class SequenceNPY ; 
   friend class RecordsNPY ; 
   friend class PhotonsNPY ; 
   friend class HitsNPY ; 
   friend class G4StepNPY ; 
   friend class MaterialLibNPY ; 

   public:
       static Type_t type ;  // for type branching 
       static T UNSET ; 

   public:
       // NB favor vec4 over vec3 for better GPU performance (due to memory coalescing/alignment)
       static NPY<T>* make_vec3(float* m2w, unsigned int npo=100);  

       static NPY<T>* make(NPYSpec* spec);
       static NPY<T>* make(std::vector<int>& shape);
       static NPY<T>* make(unsigned int ni);
       static NPY<T>* make(unsigned int ni, unsigned int nj );
       static NPY<T>* make(unsigned int ni, unsigned int nj, unsigned int nk );
       static NPY<T>* make(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl );
       static NPY<T>* make(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, unsigned int nm);

       static NPY<T>* make_modulo(NPY<T>* src, unsigned int scaledown);
       static NPY<T>* make_repeat(NPY<T>* src, unsigned int n);
       static NPY<T>* make_like(NPY<T>* src);      // same shape as source, zeroed
       static NPY<T>* make_dbg_like(NPY<T>* src, int label_=0);  // same shape as source, values based on indices controlled with label_


       // ctor takes ownership of a copy of the inputs 
       NPY(std::vector<int>& shape, T*  data            , std::string& metadata) ;
       NPY(std::vector<int>& shape, std::vector<T>& data, std::string& metadata) ;

   public:
       static NPY<T>* debugload(const char* path);
       static NPY<T>* load(const char* path, bool quietly=false);
       static NPY<T>* load(const char* dir, const char* name, bool quietly=false);
       //static NPY<T>* load(const char* typ, const char* tag, const char* det, bool quietly=false);
       static NPY<T>* load(const char* tfmt, const char* targ, const char* tag, const char* det, bool quietly=false); 
 
       void save(const char* path);
       void save(const char* dir, const char* name);
       //void save(const char* typ, const char* tag, const char* det );
       void save(const char* tfmt, const char* targ, const char* tag, const char* det);


       bool exists(const char* path);
       bool exists(const char* dir, const char* name);
       //bool exists(const char* typ, const char* tag, const char* det);
       bool exists(const char* tfmt, const char* targ, const char* tag, const char* det );

       // manipulations change types, not tags:  tfmt % targ -> typ
   public:
       NPY<T>* clone();
       static NPY<T>* copy(NPY<T>* src);
       NPY<T>* make_slice(const char* slice);
       NPY<T>* make_slice(NSlice* slice);
   public:
       NPY<T>* transform(glm::mat4& tr);
       NPY<T>* scale(float factor);
   public:
       T maxdiff(NPY<T>* other, bool dump=false);
   public:
       T* getValues();
       //unsigned int getNumValues(); tis in base class
       T* begin();
       T* end();

       T* getValues(unsigned int i, unsigned int j=0);
       void* getBytes();
       void* getPointer();   // aping GBuffer for easier migration
       BBufSpec* getBufSpec();

       void read(void* ptr);
    private:
       T* grow(unsigned int nitems); // increase size to contain an extra nitems, return pointer to start of them
    public:
       void add(NPY<T>* other);      // add another buffer, it must have same itemsize (ie size after 1st dimension)
       void add(const T* values, unsigned int nvals);   // add values, nvals must be integral multiple of the itemsize  
       void add(void* bytes, unsigned int nbytes); // add bytes,  nbytes must be integral multiple of itemsize in bytes
       void add(T x, T y, T z, T w) ;   // add values of a quad, itemsize must be 4 
       void add(const glm::vec4& v ) ;  // add quad, itemsize must be 4 
    public:
       std::vector<T>& data();
       void setData(T* data);
       void fill(T value);
       T* zero();
       T* allocate();
    public:
       T* getUnsetItem();
       bool isUnsetItem(unsigned int i, unsigned int j);
    public:
       void dump(const char* msg="NPY::dump", unsigned int limit=15);
    public:
       // methods assuming 3D shape
       std::set<int> uniquei(unsigned int j, unsigned int k);

       std::map<int,int> count_uniquei(unsigned int j, unsigned int k, int sj=-1, int sk=-1);
       // when both sj and sk are >-1 the float specified is used 
       // to sign the boundary code used in the map 

       std::map<unsigned int,unsigned int> count_unique_u(unsigned int j, unsigned int k);

       std::vector<std::pair<int,int> > count_uniquei_descending(unsigned int j, unsigned int k); 
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);

    public:
       // type shifting get/set using union trick

       unsigned int getUSum(unsigned int j, unsigned int k);

       T            getValue(unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);
       float        getFloat(unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);
       unsigned int getUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);
       int          getInt(  unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);

       void         getU( short& value, unsigned short& uvalue, unsigned char& msb, unsigned char& lsb, unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);

       ucharfour    getUChar4( unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1 );
       charfour     getChar4( unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1 );

       void         setValue(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T value);
       void         setFloat(unsigned int i, unsigned int j, unsigned int k, unsigned int l, float value);
       void         setUInt( unsigned int i, unsigned int j, unsigned int k, unsigned int l, unsigned int value);
       void         setInt(  unsigned int i, unsigned int j, unsigned int k, unsigned int l, int value);

       ///  quad setters 
       void         setQuad(      const nvec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );
       void         setQuad(const   glm::vec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );
       void         setQuad(const  glm::ivec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );
       void         setQuad(const  glm::uvec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );

       void         setQuad(unsigned int i, unsigned int j,                 float x, float y=0.f, float z=0.f, float w=0.f );
       void         setQuad(unsigned int i, unsigned int j, unsigned int k, float x, float y=0.f, float z=0.f, float w=0.f );

       void         setQuadI(const glm::ivec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );
       void         setQuadU(const glm::uvec4& vec, unsigned int i, unsigned int j=0, unsigned int k=0 );

       ///  quad getters
       glm::vec4    getQuad(unsigned int i,  unsigned int j=0, unsigned int k=0 );
       glm::ivec4   getQuadI(unsigned int i, unsigned int j=0, unsigned int k=0 );
       glm::uvec4   getQuadU(unsigned int i, unsigned int j=0, unsigned int k=0 );

       glm::mat4    getMat4(unsigned int i);

   //private:
   public:
       std::vector<T>     m_data ; 
       T*                 m_unset_item ; 
       BBufSpec*          m_bufspec ; 
 
};





