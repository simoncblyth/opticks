output_event_directory_control_by_envvar
=========================================


tds3ip events clesarly belong on a different tree

::

    1929 void OpticksEvent::saveHitData(NPY<float>* ht) const
    1930 {
    1931     if(ht)
    1932     {
    1933         unsigned num_hit = ht->getNumItems();
    1934         ht->save(m_pfx, "ht", m_typ,  m_tag, m_udet);  // even when zero hits
    1935         LOG(LEVEL)
    1936              << " num_hit " << num_hit
    1937              << " ht " << ht->getShapeString()
    1938              << " tag " << m_tag
    1939              ;
    1940     }
    1941 }

    1991 void OpticksEvent::saveGenstepData()
    1992 {
    1993     // genstep were formally not saved as they exist already elsewhere,
    1994     // however recording the gs in use for posterity makes sense
    1995     // 
    1996     NPY<float>* gs = getGenstepData();
    1997     if(gs) gs->save(m_pfx, "gs", m_typ,  m_tag, m_udet);
    1998 }
    1999 void OpticksEvent::savePhotonData()
    2000 {
    2001     NPY<float>* ox = getPhotonData();
    2002     if(ox) ox->save(m_pfx, "ox", m_typ,  m_tag, m_udet);
    2003 }


Nasty, still using NPY functionality (that should not really belong in NPY package)::

    1089 template <typename T>
    1090 void NPY<T>::save(const char* pfx, const char* tfmt, const char* source, const char* tag, const char* det) const
    1091 {
    1092     std::string path_ = BOpticksEvent::path(pfx, det, source, tag, tfmt );
    1093     save(path_.c_str());
    1094 }
    1095 






