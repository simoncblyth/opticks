reviveAnimator
================

optickscore::


     879 void Composition::initAnimator()
     880 {
     881     float* target = glm::value_ptr(m_param) + 3 ;   // offset to ".w" 
     882 
     883 
     884 #ifdef OLD_ANIM
     885     m_animator = new Animator(target, m_animator_period, m_domain_time.x, m_domain_time.z );
     886 #else
     887     glm::vec4 animtimerange(0.f, m_domain_time.y, 0.f, 0.f) ;
     888     m_ok->getAnimTimeRange(animtimerange);
     889 
     890     float tmin = animtimerange.x < 0.f ? m_domain_time.x : animtimerange.x ;
     891     float tmax = animtimerange.y < 0.f ? m_domain_time.y : animtimerange.y ;
     892 
     893     m_animator = new Animator(target, m_animator_period, tmin , tmax, "Composition::initAnimator" );
     894 #endif
     895     m_animator->setModeRestrict(Animator::FAST);
     896     m_animator->Summary("Composition::gui setup Animation");
     897 }
     898 
     899 
     900 Animator* Composition::getAnimator()
     901 {
     902     if(!m_animator) initAnimator() ;
     903     return m_animator ;
     904 }
     905 
     906 
     907 void Composition::initRotator()
     908 {
     909     m_rotator = new Animator(&m_lookphi, 180, -180.f, 180.f, "Composition::initRotator");
     910     m_rotator->setModeRestrict(Animator::NORM);  // only OFF and SLOW 
     911     m_rotator->Summary("Composition::initRotator");
     912 }
     913 
     914 
     915 void Composition::nextAnimatorMode(unsigned modifiers)
     916 {
     917     if(!m_animator) initAnimator() ;
     918     m_animator->nextMode(modifiers);
     919 }
     920 


::

    1990 void Composition::update()
    1991 {
    ....
    2026     float pi = boost::math::constants::pi<float>() ;
    2027     m_lookrotation = glm::rotate(glm::mat4(1.f), m_lookphi*float(pi)/180.f , Y );
    2028     m_ilookrotation = glm::transpose(m_lookrotation);


    2045     m_trackball->getOrientationMatrices(m_trackballrot, m_itrackballrot);  // this is just rotation, no translation
    2046     m_trackball->getTranslationMatrices(m_trackballtra, m_itrackballtra);  // just translation  
    2047 
    2048     m_world2eye = m_trackballtra * m_look2eye * m_trackballrot * m_lookrotation * m_eye2look * m_world2camera ;           // ModelView
    2049 
    2050     m_eye2world = m_camera2world * m_look2eye * m_ilookrotation * m_itrackballrot * m_eye2look * m_itrackballtra ;          // InverseModelView




