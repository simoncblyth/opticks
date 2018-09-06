#include <cstddef>
#include "GProperty.hh"
#include "X4Property.hh"


template <typename T>
G4PhysicsVector* X4Property<T>::Convert(const GProperty<T>* prop) 
{
    X4Property<T> xprop(prop); 
    return xprop.getVector(); 
}

template <typename T>
G4PhysicsVector* X4Property<T>::getVector() const { return m_vec ; }


template <typename T>
X4Property<T>::X4Property( const GProperty<T>* prop )
    :
    m_prop(prop),
    m_vec(NULL)
{
    init(); 
}


template <typename T>
void X4Property<T>::init()
{
    unsigned nval = m_prop->getLength();

    




}




template class X4Property<float>;
template class X4Property<double>;

