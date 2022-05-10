
void QSim::cerenkov_photon( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}

template<typename T>
void QSim::cerenkov_photon_enprop( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon_enprop<T>(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}

template void QSim::cerenkov_photon_enprop<float>(  quad4*, unsigned ); 
template void QSim::cerenkov_photon_enprop<double>( quad4*, unsigned ); 

void QSim::cerenkov_photon_expt( quad4* photon, unsigned num_photon )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 
    QSim_cerenkov_photon_expt(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon );  
    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 
}



extern void QSim_cerenkov_wavelength_rejection_sampled(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, float* wavelength, unsigned num_wavelength ); 

NP* QSim::cerenkov_wavelength_rejection_sampled(unsigned num_wavelength )
{
    LOG(LEVEL) << "[ num_wavelength " << num_wavelength ;
 
    configureLaunch(num_wavelength, 1 ); 

    float* d_wavelength = QU::device_alloc<float>(num_wavelength); 

    QSim_cerenkov_wavelength_rejection_sampled(numBlocks, threadsPerBlock, d_sim, d_wavelength, num_wavelength );  

    NP* w = NP::Make<float>(num_wavelength) ; 

    QU::copy_device_to_host_and_free<float>( (float*)w->bytes(), d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 

    return w ; 
}




extern void QSim_cerenkov_generate(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );

template <typename T>
extern void QSim_cerenkov_generate_enprop(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );

extern void QSim_cerenkov_generate_expt(dim3 numBlocks, dim3 threadsPerBlock, qsim* d_sim, quad4* photon, unsigned num_photon );



NP* QSim::cerenkov_generate(unsigned num_photon, unsigned test )
{
    configureLaunch(num_photon, 1 ); 
    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 

    switch(test)
    {
        case CERENKOV_GENERATE:                QSim_cerenkov_generate(              numBlocks,  threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_GENERATE_ENPROP_FLOAT:   QSim_cerenkov_generate_enprop<float>(numBlocks,  threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_GENERATE_ENPROP_DOUBLE:  QSim_cerenkov_generate_enprop<double>(numBlocks, threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
        case CERENKOV_GENERATE_EXPT:           QSim_cerenkov_generate_expt(          numBlocks, threadsPerBlock, d_sim, d_photon, num_photon ); break ; 
    }

    NP* p = NP::Make<float>( num_photon, 4, 4); 
    quad4* pp = (quad4*)p->bytes() ; 
    QU::copy_device_to_host_and_free<quad4>( pp, d_photon, num_photon ); 
    return p ; 
}



