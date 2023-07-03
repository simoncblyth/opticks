#pragma once
/**
U4PhysicsTable.h 
==================

Canonical usage is at U4Tree instanciation 
within U4Tree::initRayleigh 

HMM: trying a different approach from former impl
for handling the Geant4 Water RAYLEIGH from RINDEX 
special casing. Former approach was material centric::

   X4MaterialWater.hh
   X4MaterialWater.cc
   X4OpRayleigh.hh
   X4OpRayleigh.cc

Instead of focussing on Water material and rayleigh process
just grab the entire physics table from a templated process.
So can peruse the physics table and use it as appropriate. 

"g4-cls G4PhysicsTable" ISA : std::vector<G4PhysicsVector*>


When called without a process argument U4PhysicsTable
attempts access the process from process manager. 
This approach requires this to be run after physics setup.::

    NP* tab = U4PhysicsTable<G4OpRayleigh>::Convert();  

Directly passing a process instance gives more 
flexibility, allowing running from U4Tree::initRayleigh::

    NP* tab = U4PhysicsTable<G4OpRayleigh>::Convert(new G4OpRayleigh); 


**/

#include <string>
#include <sstream>
#include <iostream>

#include "U4Process.h"
#include "U4PhysicsVector.h"

template<typename T>
struct U4PhysicsTable
{
    T*              proc ; 
    G4PhysicsTable* table ; 
    NP*             tab ; 

    static NP* Convert(T* proc_=nullptr); 
    U4PhysicsTable(T* proc_=nullptr); 

    std::string desc() const ; 
};

template<typename T>
inline NP* U4PhysicsTable<T>::Convert(T* proc)
{
    U4PhysicsTable<T> table(proc) ; 
    if(table.tab == nullptr) std::cerr << table.desc() ; 
    return table.tab ; 
}

template<typename T>
inline U4PhysicsTable<T>::U4PhysicsTable(T* proc_)
    :
    proc(proc_ ? proc_ : U4Process::Get<T>()),
    table(proc ? proc->GetPhysicsTable() : nullptr),
    tab(table ? U4PhysicsVector::CreateCombinedArray(table) : nullptr)
{
}

template<typename T>
inline std::string U4PhysicsTable<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "U4PhysicsTable::desc"
        << " proc " << ( proc ? "YES" : "NO " ) << std::endl 
        << " procName " << ( proc ? proc->GetProcessName() : "-" ) << std::endl 
        << " table " << ( table ? "YES" : "NO " ) << std::endl 
        << " tab " << ( tab ? tab->sstr() : "-" ) << std::endl 
        ;
    std::string str = ss.str(); 
    return str ; 
}

