#include <sstream>

#include "PLOG.hh"
#include "NPY.hpp"
#include "SensorLib.hh"

const plog::Severity SensorLib::LEVEL = PLOG::EnvLevel("SensorLib", "DEBUG"); 

const char* SensorLib::SENSOR_DATA = "sensorData.npy" ;
const char* SensorLib::SENSOR_ANGULAR_EFFICIENCY = "angularEfficiency.npy" ;

SensorLib* SensorLib::Load(const char* dir)  // static 
{
    LOG(info) << dir ; 
    SensorLib* sensorlib = new SensorLib(dir) ; 
    return sensorlib  ; 
}

SensorLib::SensorLib(const char* dir)
    :
    m_loaded(dir ? true : false),
    m_sensor_data(m_loaded ? NPY<float>::load(dir, SENSOR_DATA) :  NULL),
    m_sensor_num(m_loaded && m_sensor_data != NULL ? m_sensor_data->getNumItems() : 0 ),
    m_sensor_angular_efficiency(m_loaded ? NPY<float>::load(dir, SENSOR_ANGULAR_EFFICIENCY) : NULL),
    m_closed(false)
{
    LOG(LEVEL);
}


unsigned SensorLib::getNumSensor() const 
{
    return m_sensor_num ; 
}

void SensorLib::save(const char* dir) const 
{
    LOG(info) << dir ; 
    if(m_sensor_data != NULL)
        m_sensor_data->save(dir, SENSOR_DATA); 

    if(m_sensor_angular_efficiency != NULL)
        m_sensor_angular_efficiency->save(dir, SENSOR_ANGULAR_EFFICIENCY );
}


std::string SensorLib::desc() const 
{
    unsigned num_category = getNumSensorCategories(); // 0 when no 
    std::stringstream ss ; 
    ss
       << "SensorLib"
       << " closed " << ( m_closed ? "Y" : "N" ) 
       << " loaded " << ( m_loaded ? "Y" : "N" ) 
       << " sensor_data " << ( m_sensor_data ? m_sensor_data->getShapeString() : "N" )
       << " sensor_num " << m_sensor_num        
       << " sensor_angular_efficiency " << ( m_sensor_angular_efficiency ? m_sensor_angular_efficiency->getShapeString() : "N" )
       << " num_category " << num_category 
       ;
    return ss.str(); 
}






void SensorLib::dump(const char* msg) const 
{
    dumpSensorData(msg);
    dumpAngularEfficiency(msg);
}

void SensorLib::dumpSensorData(const char* msg) const 
{
    LOG(info) << msg ; 
    LOG(info) << desc() ; 

    float efficiency_1 ; 
    float efficiency_2 ; 
    int   category ; 
    int   identifier ; 

    int w = 12 ; 

    std::cout 
        << std::setw(w) << "sensorIndex" 
        << " : "
        << std::setw(w) << "efficiency_1" 
        << " : "
        << std::setw(w) << "efficiency_2"
        << " : "
        << std::setw(w) << "category"
        << " : "
        << std::setw(w) << "identifier"
        << std::endl 
        ;
    
    for(unsigned i=0 ; i < m_sensor_num ; i++)
    {
        unsigned sensorIndex = i ; 
        getSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);
        std::cout 
            << std::setw(w) << sensorIndex 
            << " : "
            << std::setw(w) << efficiency_1 
            << " : "
            << std::setw(w) << efficiency_2
            << " : "
            << std::setw(w) << category
            << " : "
            << std::setw(w) << identifier
            << std::endl 
            ;
    }  
}


/**
SensorLib::initSensorData
---------------------------

Canonically invoked by G4Opticks::setGeometry

**/

void SensorLib::initSensorData(unsigned sensor_num)
{
    assert( ! m_loaded ) ; 
    LOG(LEVEL) << " sensor_num " << sensor_num  ;
    m_sensor_num = sensor_num ;  
    m_sensor_data = NPY<float>::make(m_sensor_num, 4);  
    m_sensor_data->zero(); 
}

/**
SensorLib::setSensorData
---------------------------

Calls to this for all sensor_placements G4PVPlacement provided by SensorLib::getSensorPlacements
provides a way to associate the Opticks contiguous 0-based sensorIndex with a detector 
defined sensor identifier. 

Within JUNO simulation framework this is used from LSExpDetectorConstruction::SetupOpticks.

sensorIndex 
    0-based continguous index used to access the sensor data, 
    the index must be less than the number of sensors
efficiency_1 
efficiency_2
    two efficiencies which are multiplied together with the local angle dependent efficiency 
    to yield the detection efficiency used to assign SURFACE_COLLECT to photon hits 
    that already have SURFACE_DETECT 
category
    used to distinguish between sensors with different theta textures   
identifier
    detector specific integer representing a sensor, does not need to be contiguous


**/

void SensorLib::setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int category, int identifier)
{
    assert( sensorIndex < m_sensor_num );
    m_sensor_data->setFloat(sensorIndex,0,0,0, efficiency_1);
    m_sensor_data->setFloat(sensorIndex,1,0,0, efficiency_2);
    m_sensor_data->setInt(  sensorIndex,2,0,0, category);
    m_sensor_data->setInt(  sensorIndex,3,0,0, identifier);
}

void SensorLib::getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const
{   
    assert( sensorIndex < m_sensor_num ); 
    assert( m_sensor_data );
    efficiency_1 = m_sensor_data->getFloat(sensorIndex,0,0,0);
    efficiency_2 = m_sensor_data->getFloat(sensorIndex,1,0,0);
    category     = m_sensor_data->getInt(  sensorIndex,2,0,0);
    identifier   = m_sensor_data->getInt(  sensorIndex,3,0,0);
}

int SensorLib::getSensorIdentifier(unsigned sensorIndex) const
{
    assert( sensorIndex < m_sensor_num );
    assert( m_sensor_data );
    return m_sensor_data->getInt( sensorIndex, 3, 0, 0);
}

/*
template <typename T>
void SensorLib::setSensorDataMeta( const char* key, T value )
{
    assert( m_sensor_data );
    m_sensor_data->setMeta<T>( key, value );
}
*/

void SensorLib::setSensorAngularEfficiency( 
        const std::vector<int>& shape, 
        const std::vector<float>& values,
        int theta_steps, float theta_min, float theta_max,
        int phi_steps,   float phi_min, float phi_max )
{
    LOG(LEVEL) << "[" ;
    const NPY<float>* a = MakeSensorAngularEfficiency(shape, values, theta_steps, theta_min, theta_max, phi_steps, phi_min, phi_max) ;
    setSensorAngularEfficiency(a);
    LOG(LEVEL) << "]" ;
}

const NPY<float>*  SensorLib::MakeSensorAngularEfficiency(        // static 
          const std::vector<int>& shape, 
          const std::vector<float>& values,
          int theta_steps, float theta_min, float theta_max,  
          int phi_steps,   float phi_min, float phi_max )   
{
    std::string metadata = "" ;
    NPY<float>* a = new NPY<float>(shape, values, metadata);
    a->setMeta<int>("theta_steps", theta_steps);
    a->setMeta<float>("theta_min", theta_min);
    a->setMeta<float>("theta_max", theta_max);
    a->setMeta<int>("phi_steps", phi_steps);
    a->setMeta<float>("phi_min", phi_min);
    a->setMeta<float>("phi_max", phi_max);
    return a ;
}


void SensorLib::setSensorAngularEfficiency( const NPY<float>* sensor_angular_efficiency )
{
    m_sensor_angular_efficiency = sensor_angular_efficiency ;
}

unsigned SensorLib::getNumSensorCategories() const
{
    return m_sensor_angular_efficiency ? m_sensor_angular_efficiency->getShape(0) : 0 ; 
} 


void SensorLib::dumpAngularEfficiency(const char* msg) const 
{
    LOG(info) << msg ; 

    unsigned num_dimensions = m_sensor_angular_efficiency->getNumDimensions(); 
    assert( num_dimensions == 4 ); 

    unsigned ni = m_sensor_angular_efficiency->getShape(0); 
    unsigned nj = m_sensor_angular_efficiency->getShape(1); 
    unsigned nk = m_sensor_angular_efficiency->getShape(2); 
    unsigned nl = m_sensor_angular_efficiency->getShape(3); 

    unsigned num_cat = ni ; 
    unsigned num_theta = nj ; 
    unsigned num_phi   = nk ; 
    unsigned num_elem = nl ;   // multiplicity 


    LOG(info) 
        << " num_cat " << num_cat  
        << " num_theta " << num_theta  
        << " num_phi " << num_phi  
        << " num_elem " << num_elem  
        ;

    assert( num_elem == 1 ); 

    unsigned edgeitems = 8 ; 
    unsigned w = 8 ; 

    std::stringstream ss ; 


    ss << " " << std::setw(3) << "" << "  " ; 
    for(unsigned k=0 ; k < nk ; k++)
    {
        if( k < edgeitems || k > nk - edgeitems ) ss << std::setw(w) << k << " " ; 
        else if( k == edgeitems )  ss << std::setw(w) << "..." << " " ;
    }
    std::string phi_labels = ss.str(); 


    for(unsigned i=0 ; i < ni ; i++)
    {
        std::cout << " category " << i << std::endl ; 
        std::cout << phi_labels << std::endl ; 
 
        for(unsigned j=0 ; j < nj ; j++)
        {
            std::cout << "(" << std::setw(3) << j << ") " ; 

            for(unsigned k=0 ; k < nk ; k++)
            {
                float value = m_sensor_angular_efficiency->getValue(i, j, k);             

                if( k < edgeitems || k > nk - edgeitems )
                {
                    std::cout << std::setw(w) << value << " "  ; 
                }
                else if( k == edgeitems )
                {
                    std::cout << std::setw(w) << "..." << " "  ; 
                }
            }
            std::cout << std::endl ; 
        }
        std::cout << phi_labels << std::endl ; 
    } 
}


NPY<float>*  SensorLib::getSensorDataArray() const
{
    return m_sensor_data ;
}
const NPY<float>*  SensorLib::getSensorAngularEfficiencyArray() const
{
    return m_sensor_angular_efficiency ;
}



bool SensorLib::isClosed() const 
{
    return m_closed ; 
}


/**
SensorLib::close
-----------------

Closing the sensorlib checks consistency between the 
sensorData and angularEfficiency arrays.  

The 0-based category index from the sensorData must be less than the number 
of angularEfficiency categories when an angularEfficiency array is present.
When no angularEfficiency array has been set the sensorData categories 
must all be -1.

**/


void SensorLib::close() 
{
    if(m_closed) 
    {
        LOG(error) << " closed already " ;
        return ;   
    }

    if(m_sensor_num == 0 ) 
    {
        LOG(error) << " SKIP as m_sensor_num zero " ;
        return ;   
    }

    bool dump = true ; 
    checkSensorCategories(dump); 

    m_closed = true ; 
    LOG(info) << desc() ; 
}

void SensorLib::checkSensorCategories(bool dump)
{
    LOG(info) << "[ " <<  desc() ; 
    unsigned num_category = getNumSensorCategories(); // 0 when none
 
    m_category_counts.clear(); 

    for(unsigned i=0 ; i < m_sensor_num ; i++)
    {
        unsigned sensorIndex = i ; 

        float efficiency_1 ; 
        float efficiency_2 ; 
        int category ; 
        int identifier ; 

        getSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);

        bool category_expected = ( num_category == 0 ) ? category == -1 : category > -1 ; 
        if(!category_expected || dump)
        std::cout 
            << " sensorIndex "  << std::setw(6) << sensorIndex
            << " efficiency_1 " << std::setw(10) << efficiency_1 
            << " efficiency_2 " << std::setw(10) << efficiency_2 
            << " category "     << std::setw(6) << category
            << " identifier "   << std::setw(10) << std::hex << identifier << std::dec
            << " category_expected " << ( category_expected ? "Y" : "N" ) 
            << std::endl 
            ;
        assert(category_expected); 
        m_category_counts[category] += 1 ;  
    }

    dumpCategoryCounts("SensorLib::checkSensorCategories"); 

    LOG(info) << "] " <<  desc() ; 
}

void SensorLib::dumpCategoryCounts(const char* msg) const 
{
    LOG(info) << msg ; 
    typedef std::map<int,int>::const_iterator IT ;
    for(IT it=m_category_counts.begin() ; it != m_category_counts.end() ; it++ )
    {
        std::cout
            << " category " << std::setw(10) << it->first 
            << " count "    << std::setw(10) << it->second
            << std::endl 
            ;
    }
}



/*
template <typename T>
void SensorLib::setSensorAngularEfficiencyMeta( const char* key, T value )
{
    assert( m_sensor_angular_efficiency ); 
    m_sensor_angular_efficiency->setMeta<T>( key, value ); 
}

template OKGEO_API void SensorLib::setSensorDataMeta(const char* key, int value);
template OKGEO_API void SensorLib::setSensorDataMeta(const char* key, float value);
template OKGEO_API void SensorLib::setSensorDataMeta(const char* key, std::string value);

template OKGEO_API void SensorLib::setSensorAngularEfficiencyMeta(const char* key, int value);
template OKGEO_API void SensorLib::setSensorAngularEfficiencyMeta(const char* key, float value);
template OKGEO_API void SensorLib::setSensorAngularEfficiencyMeta(const char* key, std::string value);
*/

