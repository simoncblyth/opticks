/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "SStr.hh"
#include "NGPU.hpp"
#include "NPY.hpp"
#include "PLOG.hh"

NGPU* NGPU::fInstance = NULL ; 

NGPU* NGPU::GetInstance()
{
    if( fInstance == NULL ) fInstance = new NGPU ; 
    return fInstance ; 
}

NGPU::NGPU() 
   :
   recs(NPY<ULL>::make(0,4))
{
}

NGPU::NGPU(NPY<ULL>* recs_) 
   :
   recs(recs_)
{
    import(); 
}

NGPU* NGPU::Load(const char* path)
{
    NPY<ULL>* recs = NPY<ULL>::load(path) ;  
    return recs ? new NGPU(recs) : NULL ; 
}

void NGPU::saveBuffer(const char* path)
{
    recs->save(path); 
}

void NGPU::add( unsigned long long num_bytes, const char* name, const char* owner, const char* note )
{
    ULL name_ = SStr::ToULL(name) ;
    ULL owner_ = SStr::ToULL(owner) ;
    ULL note_ = SStr::ToULL(note) ;

    recs->add( name_, owner_, note_, num_bytes ); 

    NGPURecord rec( name_, owner_ , note_ , num_bytes ); 
    records.push_back(rec); 
}

void NGPURecord::set_name(const char* name_){ strncpy(name, name_, 8); }
void NGPURecord::set_owner(const char* owner_){ strncpy(owner, owner_, 8); }
void NGPURecord::set_note(const char* note_){ strncpy(note, note_, 8); }

NGPURecord::NGPURecord( 
         unsigned long long name_, 
         unsigned long long owner_, 
         unsigned long long note_, 
         unsigned long long num_bytes_) 
     :
     num_bytes(num_bytes_)
{
    SStr::FillFromULL( name, name_ );  
    SStr::FillFromULL( owner, owner_ );  
    SStr::FillFromULL( note, note_ );  
}

std::string NGPURecord::desc() const  
{
    std::stringstream ss ; 
    ss << std::setw(15) << name
       << std::setw(15) << owner
       << std::setw(15) << note
       << " : " 
       << std::setw(15) << num_bytes 
       << " : " 
       << std::setw(10) << std::fixed << std::setprecision(2) << float(num_bytes)/1e6f
       ;
    return ss.str(); 
}

void NGPU::import()
{
    for(unsigned i=0 ; i < recs->getNumItems() ; i++)
    {
        ULL name_ = recs->getValue(i, 0, 0) ; 
        ULL owner_ = recs->getValue(i, 1, 0) ; 
        ULL note_ = recs->getValue(i,  2, 0) ;
        ULL num_bytes = recs->getValue(i, 3, 0 ) ;  

        NGPURecord rec( name_, owner_, note_, num_bytes ) ; 
        //LOG(info) << rec.desc() ; 
        records.push_back(rec); 
    }
}

unsigned long long NGPU::getNumBytes() const 
{
    unsigned num_records = records.size() ;
    unsigned long long num_bytes(0) ; 
    for(unsigned i=0 ; i < num_records ; i++)
    {
        const NGPURecord& rec = records[i] ; 
        num_bytes += rec.num_bytes ; 
    }
    return num_bytes ; 
}


void NGPU::dump(const char* msg) const 
{
    unsigned num_records = records.size() ;
    unsigned num_bytes = getNumBytes() ; 

    LOG(info) << msg 
              << " num_records " << num_records
              << " num_bytes " << num_bytes
             ; 

    for(unsigned i=0 ; i < num_records ; i++)
    {
        const NGPURecord& rec = records[i] ; 
        std::cout 
             << std::setw(3) << i 
             << " : " 
             << rec.desc() 
             << std::endl
             ; 
    }

    std::cout << std::endl ; 
    std::cout 
              << std::setw(3) << num_records
              << std::setw(15) << "" 
              << std::setw(30) << "<- records TOTALS bytes, Mbytes -> " 
              << " : " 
              << std::setw(15) << num_bytes 
              << " : " 
              << std::setw(10) << std::fixed << std::setprecision(2) << float(num_bytes)/1e6f
              << std::endl ; 

}



