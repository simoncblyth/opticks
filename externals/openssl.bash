
openssl-env(){      opticks- ;  }
openssl-vi(){       vi $BASH_SOURCE ; }
openssl-usage(){ cat << EOU

EOU
}

openssl-info(){ cat << EOI

   openssl-dir    : $(openssl-dir)
   openssl-bdir   : $(openssl-bdir)
   openssl-idir   : $(openssl-idir)
   openssl-prefix : $(openssl-prefix)

EOI
}


openssl-dir(){  echo $(opticks-prefix)/externals/openssl/$(openssl-name) ; }
openssl-bdir(){ echo $(openssl-dir).build ; }

#openssl-idir(){ echo $(opticks-prefix)/externals ; }
openssl-idir(){ echo /tmp/$USER/opticks/externals/openssl/$(openssl-name) ; }

openssl-prefix(){ echo $(openssl-idir) ; }

openssl-cd(){  cd $(openssl-dir); }
openssl-bcd(){ cd $(openssl-bdir); }
openssl-icd(){ cd $(openssl-idir); }

openssl-version(){ echo 3.2.0 ; }
openssl-name(){ echo openssl-$(openssl-version) ; }  # NB no lib
openssl-url(){
   echo https://github.com/openssl/openssl/releases/download/openssl-3.2.0/$(openssl-name).tar.gz
}


openssl-get()
{
   local dir=$(dirname $(openssl-dir)) &&  mkdir -p $dir && cd $dir

   local rc=0
   local url=$(openssl-url)
   local tgz=$(basename $url)
   local opt=$( [ -n "${VERBOSE}" ] && echo "-xzf" || echo "-xzvf" )

   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && opticks-curl $url
   [ ! -d "$nam" ] && tar $opt $tgz
   [ ! -d "$nam" ] && rc=1

   return $rc
}

openssl-build()
{
    openssl-cd
    ./config \
       --prefix=$(openssl-prefix) \
       --openssldir=$(openssl-prefix)/ssl \
       shared \
       zlib

   #  -Wl,-rpath,'$(LIBRPATH)'


    # Build (OpenSSL 3.x is significantly faster with parallel jobs)
    make -j$(nproc)

    # Optional but recommended: run tests (takes a few minutes)
    # make test

    make install
}



openssl--()
{
   local msg="=== $FUNCNAME :"
   openssl-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1

   openssl-build
   [ $? -ne 0 ] && echo $msg build FAIL && return 2

   openssl-bashrc
   [ $? -ne 0 ] && echo $msg bashrc FAIL && return 2

   return 0
}


openssl-bashrc()
{
   if [ "$(openssl-prefix)" != "$(opticks-prefix)/externals" ]; then
       local path=$(openssl-prefix)/bashrc
       echo $BASH_SOURCE $FUNCNAME - writing $path - prefix needs to be named and versioned for this to make sense
       openssl-bashrc- > $path
   fi
}


openssl-bashrc-()
{
   cat << EOH
## generated $(date) by $(realpath $BASH_SOURCE) $FUNCNAME

HERE=\$(dirname \$(realpath \$BASH_SOURCE))

if [ -z "\${JUNOTOP}" ]; then
export JUNO_EXTLIB_openssl_HOME=\$HERE
else
export JUNO_EXTLIB_openssl_HOME=\$HERE
fi
EOH

cat << \EOS
export PATH=${JUNO_EXTLIB_openssl_HOME}/bin:${PATH}
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_openssl_HOME}/lib:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_openssl_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib64 ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_openssl_HOME}/lib64:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_openssl_HOME}/lib64/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_openssl_HOME}/lib64/pkgconfig:${PKG_CONFIG_PATH}
fi
export CPATH=${JUNO_EXTLIB_openssl_HOME}/include:${CPATH}
export MANPATH=${JUNO_EXTLIB_openssl_HOME}/share/man:${MANPATH}

# For CMake search path
export CMAKE_PREFIX_PATH=${JUNO_EXTLIB_openssl_HOME}:${CMAKE_PREFIX_PATH}

EOS

}





openssl-wipe(){
  local bdir=$(openssl-bdir)
  rm -rf $bdir

  openssl-wipe-manifest
}


openssl-wipe-manifest()
{
    local prefix=$(openssl-prefix)
    [ ! -d "$prefix" ] && echo "Prefix $prefix does not exist." && return

    echo "Wiping openssl from $prefix..."

    # 1. Disable literal string treatment, allow globbing
    # 2. Iterate through the manifest
    openssl-manifest | while read -r rel; do
        # Use an array to handle glob expansion
        local paths=( ${prefix}/${rel} )

        for path in "${paths[@]}"; do
            if [ -e "$path" ] || [ -L "$path" ]; then
                echo "Removing $path"
                rm -rf "$path"
            fi
        done
    done

    # Cleanup empty leaf directories
    find "$prefix" -type d -empty -delete 2>/dev/null
}

openssl-find()
{
    find $(openssl-prefix) -type f
}



openssl-manifest(){ cat << EOM
lib64/libcrypto.so.3
lib64/libssl.so.3
lib64/libcrypto.a
lib64/libssl.a
lib64/pkgconfig/libcrypto.pc
lib64/pkgconfig/libssl.pc
lib64/pkgconfig/openssl.pc
lib64/engines-3/afalg.so
lib64/engines-3/capi.so
lib64/engines-3/loader_attic.so
lib64/engines-3/padlock.so
lib64/ossl-modules/legacy.so
include/openssl
bin/openssl
bin/c_rehash
ssl
share/man/man1/CA.pl.1ossl
share/man/man1/openssl-*
share/man/man1/openssl.1ossl
share/man/man1/tsget.1ossl
share/man/man3/ADMISSIONS.3ossl
share/man/man3/ASN1_EXTERN_FUNCS.3ossl
share/man/man3/ASN1_INTEGER_get_int64.3ossl
share/man/man3/ASN1_INTEGER_new.3ossl
share/man/man3/ASN1_ITEM_lookup.3ossl
share/man/man3/ASN1_OBJECT_new.3ossl
share/man/man3/ASN1_STRING_TABLE_add.3ossl
share/man/man3/ASN1_STRING_length.3ossl
share/man/man3/ASN1_STRING_new.3ossl
share/man/man3/ASN1_STRING_print_ex.3ossl
share/man/man3/ASN1_TIME_set.3ossl
share/man/man3/ASN1_TYPE_get.3ossl
share/man/man3/ASN1_aux_cb.3ossl
share/man/man3/ASN1_generate_nconf.3ossl
share/man/man3/ASN1_item_d2i_bio.3ossl
share/man/man3/ASN1_item_new.3ossl
share/man/man3/ASN1_item_sign.3ossl
share/man/man3/ASYNC_WAIT_CTX_new.3ossl
share/man/man3/ASYNC_start_job.3ossl
share/man/man3/BF_encrypt.3ossl
share/man/man3/BIO_ADDR.3ossl
share/man/man3/BIO_ADDRINFO.3ossl
share/man/man3/BIO_connect.3ossl
share/man/man3/BIO_ctrl.3ossl
share/man/man3/BIO_f_base64.3ossl
share/man/man3/BIO_f_buffer.3ossl
share/man/man3/BIO_f_cipher.3ossl
share/man/man3/BIO_f_md.3ossl
share/man/man3/BIO_f_null.3ossl
share/man/man3/BIO_f_prefix.3ossl
share/man/man3/BIO_f_readbuffer.3ossl
share/man/man3/BIO_f_ssl.3ossl
share/man/man3/BIO_find_type.3ossl
share/man/man3/BIO_get_data.3ossl
share/man/man3/BIO_get_ex_new_index.3ossl
share/man/man3/BIO_get_rpoll_descriptor.3ossl
share/man/man3/BIO_meth_new.3ossl
share/man/man3/BIO_new.3ossl
share/man/man3/BIO_new_CMS.3ossl
share/man/man3/BIO_parse_hostserv.3ossl
share/man/man3/BIO_printf.3ossl
share/man/man3/BIO_push.3ossl
share/man/man3/BIO_read.3ossl
share/man/man3/BIO_s_accept.3ossl
share/man/man3/BIO_s_bio.3ossl
share/man/man3/BIO_s_connect.3ossl
share/man/man3/BIO_s_core.3ossl
share/man/man3/BIO_s_datagram.3ossl
share/man/man3/BIO_s_dgram_pair.3ossl
share/man/man3/BIO_s_fd.3ossl
share/man/man3/BIO_s_file.3ossl
share/man/man3/BIO_s_mem.3ossl
share/man/man3/BIO_s_null.3ossl
share/man/man3/BIO_s_socket.3ossl
share/man/man3/BIO_sendmmsg.3ossl
share/man/man3/BIO_set_callback.3ossl
share/man/man3/BIO_should_retry.3ossl
share/man/man3/BIO_socket_wait.3ossl
share/man/man3/BN_BLINDING_new.3ossl
share/man/man3/BN_CTX_new.3ossl
share/man/man3/BN_CTX_start.3ossl
share/man/man3/BN_add.3ossl
share/man/man3/BN_add_word.3ossl
share/man/man3/BN_bn2bin.3ossl
share/man/man3/BN_cmp.3ossl
share/man/man3/BN_copy.3ossl
share/man/man3/BN_generate_prime.3ossl
share/man/man3/BN_mod_exp_mont.3ossl
share/man/man3/BN_mod_inverse.3ossl
share/man/man3/BN_mod_mul_montgomery.3ossl
share/man/man3/BN_mod_mul_reciprocal.3ossl
share/man/man3/BN_new.3ossl
share/man/man3/BN_num_bytes.3ossl
share/man/man3/BN_rand.3ossl
share/man/man3/BN_security_bits.3ossl
share/man/man3/BN_set_bit.3ossl
share/man/man3/BN_swap.3ossl
share/man/man3/BN_zero.3ossl
share/man/man3/BUF_MEM_new.3ossl
share/man/man3/CMS_EncryptedData_decrypt.3ossl
share/man/man3/CMS_EncryptedData_encrypt.3ossl
share/man/man3/CMS_EnvelopedData_create.3ossl
share/man/man3/CMS_add0_cert.3ossl
share/man/man3/CMS_add1_recipient_cert.3ossl
share/man/man3/CMS_add1_signer.3ossl
share/man/man3/CMS_compress.3ossl
share/man/man3/CMS_data_create.3ossl
share/man/man3/CMS_decrypt.3ossl
share/man/man3/CMS_digest_create.3ossl
share/man/man3/CMS_encrypt.3ossl
share/man/man3/CMS_final.3ossl
share/man/man3/CMS_get0_RecipientInfos.3ossl
share/man/man3/CMS_get0_SignerInfos.3ossl
share/man/man3/CMS_get0_type.3ossl
share/man/man3/CMS_get1_ReceiptRequest.3ossl
share/man/man3/CMS_sign.3ossl
share/man/man3/CMS_sign_receipt.3ossl
share/man/man3/CMS_uncompress.3ossl
share/man/man3/CMS_verify.3ossl
share/man/man3/CMS_verify_receipt.3ossl
share/man/man3/COMP_CTX_new.3ossl
share/man/man3/CONF_modules_free.3ossl
share/man/man3/CONF_modules_load_file.3ossl
share/man/man3/CRYPTO_THREAD_run_once.3ossl
share/man/man3/CRYPTO_get_ex_new_index.3ossl
share/man/man3/CRYPTO_memcmp.3ossl
share/man/man3/CTLOG_STORE_get0_log_by_id.3ossl
share/man/man3/CTLOG_STORE_new.3ossl
share/man/man3/CTLOG_new.3ossl
share/man/man3/CT_POLICY_EVAL_CTX_new.3ossl
share/man/man3/DEFINE_STACK_OF.3ossl
share/man/man3/DES_random_key.3ossl
share/man/man3/DH_generate_key.3ossl
share/man/man3/DH_generate_parameters.3ossl
share/man/man3/DH_get0_pqg.3ossl
share/man/man3/DH_get_1024_160.3ossl
share/man/man3/DH_meth_new.3ossl
share/man/man3/DH_new.3ossl
share/man/man3/DH_new_by_nid.3ossl
share/man/man3/DH_set_method.3ossl
share/man/man3/DH_size.3ossl
share/man/man3/DSA_SIG_new.3ossl
share/man/man3/DSA_do_sign.3ossl
share/man/man3/DSA_dup_DH.3ossl
share/man/man3/DSA_generate_key.3ossl
share/man/man3/DSA_generate_parameters.3ossl
share/man/man3/DSA_get0_pqg.3ossl
share/man/man3/DSA_meth_new.3ossl
share/man/man3/DSA_new.3ossl
share/man/man3/DSA_set_method.3ossl
share/man/man3/DSA_sign.3ossl
share/man/man3/DSA_size.3ossl
share/man/man3/DTLS_get_data_mtu.3ossl
share/man/man3/DTLS_set_timer_cb.3ossl
share/man/man3/DTLSv1_get_timeout.3ossl
share/man/man3/DTLSv1_handle_timeout.3ossl
share/man/man3/DTLSv1_listen.3ossl
share/man/man3/ECDSA_SIG_new.3ossl
share/man/man3/ECDSA_sign.3ossl
share/man/man3/ECPKParameters_print.3ossl
share/man/man3/EC_GFp_simple_method.3ossl
share/man/man3/EC_GROUP_copy.3ossl
share/man/man3/EC_GROUP_new.3ossl
share/man/man3/EC_KEY_get_enc_flags.3ossl
share/man/man3/EC_KEY_new.3ossl
share/man/man3/EC_POINT_add.3ossl
share/man/man3/EC_POINT_new.3ossl
share/man/man3/ENGINE_add.3ossl
share/man/man3/ERR_GET_LIB.3ossl
share/man/man3/ERR_clear_error.3ossl
share/man/man3/ERR_error_string.3ossl
share/man/man3/ERR_get_error.3ossl
share/man/man3/ERR_load_crypto_strings.3ossl
share/man/man3/ERR_load_strings.3ossl
share/man/man3/ERR_new.3ossl
share/man/man3/ERR_print_errors.3ossl
share/man/man3/ERR_put_error.3ossl
share/man/man3/ERR_remove_state.3ossl
share/man/man3/ERR_set_mark.3ossl
share/man/man3/EVP_ASYM_CIPHER_free.3ossl
share/man/man3/EVP_BytesToKey.3ossl
share/man/man3/EVP_CIPHER_CTX_get_cipher_data.3ossl
share/man/man3/EVP_CIPHER_CTX_get_original_iv.3ossl
share/man/man3/EVP_CIPHER_meth_new.3ossl
share/man/man3/EVP_DigestInit.3ossl
share/man/man3/EVP_DigestSignInit.3ossl
share/man/man3/EVP_DigestVerifyInit.3ossl
share/man/man3/EVP_EncodeInit.3ossl
share/man/man3/EVP_EncryptInit.3ossl
share/man/man3/EVP_KDF.3ossl
share/man/man3/EVP_KEM_free.3ossl
share/man/man3/EVP_KEYEXCH_free.3ossl
share/man/man3/EVP_KEYMGMT.3ossl
share/man/man3/EVP_MAC.3ossl
share/man/man3/EVP_MD_meth_new.3ossl
share/man/man3/EVP_OpenInit.3ossl
share/man/man3/EVP_PBE_CipherInit.3ossl
share/man/man3/EVP_PKEY2PKCS8.3ossl
share/man/man3/EVP_PKEY_ASN1_METHOD.3ossl
share/man/man3/EVP_PKEY_CTX_ctrl.3ossl
share/man/man3/EVP_PKEY_CTX_get0_libctx.3ossl
share/man/man3/EVP_PKEY_CTX_get0_pkey.3ossl
share/man/man3/EVP_PKEY_CTX_new.3ossl
share/man/man3/EVP_PKEY_CTX_set1_pbe_pass.3ossl
share/man/man3/EVP_PKEY_CTX_set_hkdf_md.3ossl
share/man/man3/EVP_PKEY_CTX_set_params.3ossl
share/man/man3/EVP_PKEY_CTX_set_rsa_pss_keygen_md.3ossl
share/man/man3/EVP_PKEY_CTX_set_scrypt_N.3ossl
share/man/man3/EVP_PKEY_CTX_set_tls1_prf_md.3ossl
share/man/man3/EVP_PKEY_asn1_get_count.3ossl
share/man/man3/EVP_PKEY_check.3ossl
share/man/man3/EVP_PKEY_todata.3ossl
share/man/man3/EVP_PKEY_copy_parameters.3ossl
share/man/man3/EVP_PKEY_decapsulate.3ossl
share/man/man3/EVP_PKEY_decrypt.3ossl
share/man/man3/EVP_PKEY_derive.3ossl
share/man/man3/EVP_PKEY_digestsign_supports_digest.3ossl
share/man/man3/EVP_PKEY_encapsulate.3ossl
share/man/man3/EVP_PKEY_encrypt.3ossl
share/man/man3/EVP_PKEY_fromdata.3ossl
share/man/man3/EVP_PKEY_get_default_digest_nid.3ossl
share/man/man3/EVP_PKEY_get_field_type.3ossl
share/man/man3/EVP_PKEY_get_group_name.3ossl
share/man/man3/EVP_PKEY_get_size.3ossl
share/man/man3/EVP_PKEY_gettable_params.3ossl
share/man/man3/EVP_PKEY_is_a.3ossl
share/man/man3/EVP_PKEY_keygen.3ossl
share/man/man3/EVP_PKEY_meth_get_count.3ossl
share/man/man3/EVP_PKEY_meth_new.3ossl
share/man/man3/EVP_PKEY_new.3ossl
share/man/man3/EVP_PKEY_print_private.3ossl
share/man/man3/EVP_PKEY_set1_RSA.3ossl
share/man/man3/EVP_PKEY_set1_encoded_public_key.3ossl
share/man/man3/EVP_PKEY_set_type.3ossl
share/man/man3/EVP_PKEY_settable_params.3ossl
share/man/man3/EVP_PKEY_sign.3ossl
share/man/man3/EVP_PKEY_verify.3ossl
share/man/man3/EVP_PKEY_verify_recover.3ossl
share/man/man3/EVP_RAND.3ossl
share/man/man3/EVP_SIGNATURE.3ossl
share/man/man3/EVP_SealInit.3ossl
share/man/man3/EVP_SignInit.3ossl
share/man/man3/EVP_VerifyInit.3ossl
share/man/man3/EVP_aes_128_gcm.3ossl
share/man/man3/EVP_aria_128_gcm.3ossl
share/man/man3/EVP_bf_cbc.3ossl
share/man/man3/EVP_blake2b512.3ossl
share/man/man3/EVP_camellia_128_ecb.3ossl
share/man/man3/EVP_cast5_cbc.3ossl
share/man/man3/EVP_chacha20.3ossl
share/man/man3/EVP_des_cbc.3ossl
share/man/man3/EVP_desx_cbc.3ossl
share/man/man3/EVP_idea_cbc.3ossl
share/man/man3/EVP_md2.3ossl
share/man/man3/EVP_md4.3ossl
share/man/man3/EVP_md5.3ossl
share/man/man3/EVP_mdc2.3ossl
share/man/man3/EVP_rc2_cbc.3ossl
share/man/man3/EVP_rc4.3ossl
share/man/man3/EVP_rc5_32_12_16_cbc.3ossl
share/man/man3/EVP_ripemd160.3ossl
share/man/man3/EVP_seed_cbc.3ossl
share/man/man3/EVP_set_default_properties.3ossl
share/man/man3/EVP_sha1.3ossl
share/man/man3/EVP_sha224.3ossl
share/man/man3/EVP_sha3_224.3ossl
share/man/man3/EVP_sm3.3ossl
share/man/man3/EVP_sm4_cbc.3ossl
share/man/man3/EVP_whirlpool.3ossl
share/man/man3/HMAC.3ossl
share/man/man3/MD5.3ossl
share/man/man3/MDC2_Init.3ossl
share/man/man3/NCONF_new_ex.3ossl
share/man/man3/OBJ_nid2obj.3ossl
share/man/man3/OCSP_REQUEST_new.3ossl
share/man/man3/OCSP_cert_to_id.3ossl
share/man/man3/X509_check_ca.3ossl
share/man/man3/OCSP_request_add1_nonce.3ossl
share/man/man3/OCSP_resp_find_status.3ossl
share/man/man3/OCSP_response_status.3ossl
share/man/man3/OCSP_sendreq_new.3ossl
share/man/man3/OPENSSL_Applink.3ossl
share/man/man3/OPENSSL_FILE.3ossl
share/man/man3/OPENSSL_LH_COMPFUNC.3ossl
share/man/man3/OPENSSL_LH_stats.3ossl
share/man/man3/OPENSSL_config.3ossl
share/man/man3/OPENSSL_fork_prepare.3ossl
share/man/man3/OPENSSL_gmtime.3ossl
share/man/man3/OPENSSL_hexchar2int.3ossl
share/man/man3/OPENSSL_ia32cap.3ossl
share/man/man3/OPENSSL_init_crypto.3ossl
share/man/man3/OPENSSL_init_ssl.3ossl
share/man/man3/OPENSSL_instrument_bus.3ossl
share/man/man3/OPENSSL_load_builtin_modules.3ossl
share/man/man3/OPENSSL_malloc.3ossl
share/man/man3/OPENSSL_s390xcap.3ossl
share/man/man3/OPENSSL_secure_malloc.3ossl
share/man/man3/OPENSSL_strcasecmp.3ossl
share/man/man3/OSSL_ALGORITHM.3ossl
share/man/man3/OSSL_CALLBACK.3ossl
share/man/man3/OSSL_CMP_CTX_new.3ossl
share/man/man3/OSSL_CMP_HDR_get0_transactionID.3ossl
share/man/man3/OSSL_CMP_ITAV_new_caCerts.3ossl
share/man/man3/OSSL_CMP_ITAV_set0.3ossl
share/man/man3/OSSL_CMP_MSG_get0_header.3ossl
share/man/man3/OSSL_CMP_MSG_http_perform.3ossl
share/man/man3/OSSL_CMP_SRV_CTX_new.3ossl
share/man/man3/OSSL_CMP_STATUSINFO_new.3ossl
share/man/man3/OSSL_CMP_exec_certreq.3ossl
share/man/man3/OSSL_CMP_log_open.3ossl
share/man/man3/OSSL_CMP_validate_msg.3ossl
share/man/man3/OSSL_CORE_MAKE_FUNC.3ossl
share/man/man3/OSSL_CRMF_MSG_get0_tmpl.3ossl
share/man/man3/OSSL_CRMF_MSG_set0_validity.3ossl
share/man/man3/OSSL_CRMF_MSG_set1_regCtrl_regToken.3ossl
share/man/man3/OSSL_CRMF_MSG_set1_regInfo_certReq.3ossl
share/man/man3/OSSL_CRMF_pbmp_new.3ossl
share/man/man3/OSSL_DECODER.3ossl
share/man/man3/OSSL_DECODER_CTX.3ossl
share/man/man3/OSSL_DECODER_CTX_new_for_pkey.3ossl
share/man/man3/OSSL_DECODER_from_bio.3ossl
share/man/man3/OSSL_DISPATCH.3ossl
share/man/man3/OSSL_ENCODER.3ossl
share/man/man3/OSSL_ENCODER_CTX.3ossl
share/man/man3/OSSL_ENCODER_CTX_new_for_pkey.3ossl
share/man/man3/OSSL_ENCODER_to_bio.3ossl
share/man/man3/OSSL_ERR_STATE_save.3ossl
share/man/man3/OSSL_ESS_check_signing_certs.3ossl
share/man/man3/OSSL_HPKE_CTX_new.3ossl
share/man/man3/OSSL_HTTP_REQ_CTX.3ossl
share/man/man3/OSSL_HTTP_parse_url.3ossl
share/man/man3/OSSL_HTTP_transfer.3ossl
share/man/man3/OSSL_ITEM.3ossl
share/man/man3/OSSL_LIB_CTX.3ossl
share/man/man3/OSSL_PARAM.3ossl
share/man/man3/OSSL_PARAM_BLD.3ossl
share/man/man3/OSSL_PARAM_allocate_from_text.3ossl
share/man/man3/OSSL_PARAM_dup.3ossl
share/man/man3/OSSL_PARAM_int.3ossl
share/man/man3/OSSL_PROVIDER.3ossl
share/man/man3/OSSL_QUIC_client_method.3ossl
share/man/man3/OSSL_SELF_TEST_new.3ossl
share/man/man3/OSSL_SELF_TEST_set_callback.3ossl
share/man/man3/OSSL_STORE_INFO.3ossl
share/man/man3/OSSL_STORE_LOADER.3ossl
share/man/man3/OSSL_STORE_SEARCH.3ossl
share/man/man3/OSSL_STORE_attach.3ossl
share/man/man3/OSSL_STORE_expect.3ossl
share/man/man3/OSSL_STORE_open.3ossl
share/man/man3/OSSL_sleep.3ossl
share/man/man3/OSSL_trace_enabled.3ossl
share/man/man3/OSSL_trace_get_category_num.3ossl
share/man/man3/OSSL_trace_set_channel.3ossl
share/man/man3/OpenSSL_add_all_algorithms.3ossl
share/man/man3/OpenSSL_version.3ossl
share/man/man3/PEM_X509_INFO_read_bio_ex.3ossl
share/man/man3/PEM_bytes_read_bio.3ossl
share/man/man3/PEM_read.3ossl
share/man/man3/PEM_read_CMS.3ossl
share/man/man3/PEM_read_bio_PrivateKey.3ossl
share/man/man3/PEM_read_bio_ex.3ossl
share/man/man3/PEM_write_bio_CMS_stream.3ossl
share/man/man3/PEM_write_bio_PKCS7_stream.3ossl
share/man/man3/PKCS12_PBE_keyivgen.3ossl
share/man/man3/PKCS12_SAFEBAG_create_cert.3ossl
share/man/man3/PKCS12_SAFEBAG_get0_attrs.3ossl
share/man/man3/PKCS12_SAFEBAG_get1_cert.3ossl
share/man/man3/PKCS12_SAFEBAG_set0_attrs.3ossl
share/man/man3/PKCS12_add1_attr_by_NID.3ossl
share/man/man3/PKCS12_add_CSPName_asc.3ossl
share/man/man3/PKCS12_add_cert.3ossl
share/man/man3/PKCS12_add_friendlyname_asc.3ossl
share/man/man3/PKCS12_add_localkeyid.3ossl
share/man/man3/PKCS12_add_safe.3ossl
share/man/man3/PKCS12_create.3ossl
share/man/man3/PKCS12_decrypt_skey.3ossl
share/man/man3/PKCS12_gen_mac.3ossl
share/man/man3/PKCS12_get_friendlyname.3ossl
share/man/man3/PKCS12_init.3ossl
share/man/man3/PKCS12_item_decrypt_d2i.3ossl
share/man/man3/PKCS12_key_gen_utf8_ex.3ossl
share/man/man3/PKCS12_newpass.3ossl
share/man/man3/PKCS12_pack_p7encdata.3ossl
share/man/man3/PKCS12_parse.3ossl
share/man/man3/PKCS5_PBE_keyivgen.3ossl
share/man/man3/PKCS5_PBKDF2_HMAC.3ossl
share/man/man3/PKCS7_decrypt.3ossl
share/man/man3/PKCS7_encrypt.3ossl
share/man/man3/PKCS7_get_octet_string.3ossl
share/man/man3/PKCS7_sign.3ossl
share/man/man3/PKCS7_sign_add_signer.3ossl
share/man/man3/PKCS7_type_is_other.3ossl
share/man/man3/PKCS7_verify.3ossl
share/man/man3/PKCS8_encrypt.3ossl
share/man/man3/PKCS8_pkey_add1_attr.3ossl
share/man/man3/RAND_add.3ossl
share/man/man3/RAND_bytes.3ossl
share/man/man3/RAND_cleanup.3ossl
share/man/man3/RAND_egd.3ossl
share/man/man3/RAND_get0_primary.3ossl
share/man/man3/RAND_load_file.3ossl
share/man/man3/RAND_set_DRBG_type.3ossl
share/man/man3/RAND_set_rand_method.3ossl
share/man/man3/RC4_set_key.3ossl
share/man/man3/RIPEMD160_Init.3ossl
share/man/man3/RSA_blinding_on.3ossl
share/man/man3/RSA_check_key.3ossl
share/man/man3/RSA_generate_key.3ossl
share/man/man3/RSA_get0_key.3ossl
share/man/man3/RSA_meth_new.3ossl
share/man/man3/RSA_new.3ossl
share/man/man3/RSA_padding_add_PKCS1_type_1.3ossl
share/man/man3/RSA_print.3ossl
share/man/man3/RSA_private_encrypt.3ossl
share/man/man3/RSA_public_encrypt.3ossl
share/man/man3/RSA_set_method.3ossl
share/man/man3/RSA_sign.3ossl
share/man/man3/RSA_sign_ASN1_OCTET_STRING.3ossl
share/man/man3/RSA_size.3ossl
share/man/man3/SCT_new.3ossl
share/man/man3/SCT_print.3ossl
share/man/man3/SCT_validate.3ossl
share/man/man3/SHA256_Init.3ossl
share/man/man3/SMIME_read_ASN1.3ossl
share/man/man3/SMIME_read_CMS.3ossl
share/man/man3/SMIME_read_PKCS7.3ossl
share/man/man3/SMIME_write_ASN1.3ossl
share/man/man3/SMIME_write_CMS.3ossl
share/man/man3/SMIME_write_PKCS7.3ossl
share/man/man3/SRP_Calc_B.3ossl
share/man/man3/SRP_VBASE_new.3ossl
share/man/man3/SRP_create_verifier.3ossl
share/man/man3/SRP_user_pwd_new.3ossl
share/man/man3/SSL_CIPHER_get_name.3ossl
share/man/man3/SSL_COMP_add_compression_method.3ossl
share/man/man3/SSL_CONF_CTX_new.3ossl
share/man/man3/SSL_CONF_CTX_set1_prefix.3ossl
share/man/man3/SSL_CONF_CTX_set_flags.3ossl
share/man/man3/SSL_CONF_CTX_set_ssl_ctx.3ossl
share/man/man3/SSL_CONF_cmd.3ossl
share/man/man3/SSL_CONF_cmd_argv.3ossl
share/man/man3/SSL_CTX_add1_chain_cert.3ossl
share/man/man3/SSL_CTX_add_extra_chain_cert.3ossl
share/man/man3/SSL_CTX_add_session.3ossl
share/man/man3/SSL_CTX_config.3ossl
share/man/man3/SSL_CTX_ctrl.3ossl
share/man/man3/SSL_CTX_dane_enable.3ossl
share/man/man3/SSL_CTX_flush_sessions.3ossl
share/man/man3/SSL_CTX_free.3ossl
share/man/man3/SSL_CTX_get0_param.3ossl
share/man/man3/SSL_CTX_get_verify_mode.3ossl
share/man/man3/SSL_CTX_has_client_custom_ext.3ossl
share/man/man3/SSL_CTX_load_verify_locations.3ossl
share/man/man3/SSL_CTX_new.3ossl
share/man/man3/SSL_CTX_sess_number.3ossl
share/man/man3/SSL_CTX_sess_set_cache_size.3ossl
share/man/man3/SSL_CTX_sess_set_get_cb.3ossl
share/man/man3/SSL_CTX_sessions.3ossl
share/man/man3/SSL_CTX_set0_CA_list.3ossl
share/man/man3/SSL_CTX_set1_cert_comp_preference.3ossl
share/man/man3/SSL_CTX_set1_curves.3ossl
share/man/man3/SSL_CTX_set1_sigalgs.3ossl
share/man/man3/SSL_CTX_set1_verify_cert_store.3ossl
share/man/man3/SSL_CTX_set_alpn_select_cb.3ossl
share/man/man3/SSL_CTX_set_cert_cb.3ossl
share/man/man3/SSL_CTX_set_cert_store.3ossl
share/man/man3/SSL_CTX_set_cert_verify_callback.3ossl
share/man/man3/SSL_CTX_set_cipher_list.3ossl
share/man/man3/SSL_CTX_set_client_cert_cb.3ossl
share/man/man3/SSL_CTX_set_client_hello_cb.3ossl
share/man/man3/SSL_CTX_set_ct_validation_callback.3ossl
share/man/man3/SSL_CTX_set_ctlog_list_file.3ossl
share/man/man3/SSL_CTX_set_default_passwd_cb.3ossl
share/man/man3/SSL_CTX_set_generate_session_id.3ossl
share/man/man3/SSL_CTX_set_info_callback.3ossl
share/man/man3/SSL_CTX_set_keylog_callback.3ossl
share/man/man3/SSL_CTX_set_max_cert_list.3ossl
share/man/man3/SSL_CTX_set_min_proto_version.3ossl
share/man/man3/SSL_CTX_set_mode.3ossl
share/man/man3/SSL_CTX_set_msg_callback.3ossl
share/man/man3/SSL_CTX_set_num_tickets.3ossl
share/man/man3/SSL_CTX_set_options.3ossl
share/man/man3/SSL_CTX_set_psk_client_callback.3ossl
share/man/man3/SSL_CTX_set_quiet_shutdown.3ossl
share/man/man3/SSL_CTX_set_read_ahead.3ossl
share/man/man3/SSL_CTX_set_record_padding_callback.3ossl
share/man/man3/SSL_CTX_set_security_level.3ossl
share/man/man3/SSL_CTX_set_session_cache_mode.3ossl
share/man/man3/SSL_CTX_set_session_id_context.3ossl
share/man/man3/SSL_CTX_set_session_ticket_cb.3ossl
share/man/man3/SSL_CTX_set_split_send_fragment.3ossl
share/man/man3/SSL_CTX_set_srp_password.3ossl
share/man/man3/SSL_CTX_set_ssl_version.3ossl
share/man/man3/SSL_CTX_set_stateless_cookie_generate_cb.3ossl
share/man/man3/SSL_CTX_set_timeout.3ossl
share/man/man3/SSL_CTX_set_tlsext_servername_callback.3ossl
share/man/man3/SSL_CTX_set_tlsext_status_cb.3ossl
share/man/man3/SSL_CTX_set_tlsext_ticket_key_cb.3ossl
share/man/man3/SSL_CTX_set_tlsext_use_srtp.3ossl
share/man/man3/SSL_CTX_set_tmp_dh_callback.3ossl
share/man/man3/SSL_CTX_set_tmp_ecdh.3ossl
share/man/man3/SSL_CTX_set_verify.3ossl
share/man/man3/SSL_CTX_use_certificate.3ossl
share/man/man3/SSL_CTX_use_psk_identity_hint.3ossl
share/man/man3/SSL_CTX_use_serverinfo.3ossl
share/man/man3/SSL_SESSION_free.3ossl
share/man/man3/SSL_SESSION_get0_cipher.3ossl
share/man/man3/SSL_SESSION_get0_hostname.3ossl
share/man/man3/SSL_SESSION_get0_id_context.3ossl
share/man/man3/SSL_SESSION_get0_peer.3ossl
share/man/man3/SSL_SESSION_get_compress_id.3ossl
share/man/man3/SSL_SESSION_get_protocol_version.3ossl
share/man/man3/SSL_SESSION_get_time.3ossl
share/man/man3/SSL_SESSION_has_ticket.3ossl
share/man/man3/SSL_SESSION_is_resumable.3ossl
share/man/man3/SSL_SESSION_print.3ossl
share/man/man3/SSL_SESSION_set1_id.3ossl
share/man/man3/SSL_accept.3ossl
share/man/man3/SSL_accept_stream.3ossl
share/man/man3/SSL_alert_type_string.3ossl
share/man/man3/SSL_alloc_buffers.3ossl
share/man/man3/SSL_check_chain.3ossl
share/man/man3/SSL_clear.3ossl
share/man/man3/SSL_connect.3ossl
share/man/man3/SSL_do_handshake.3ossl
share/man/man3/SSL_export_keying_material.3ossl
share/man/man3/SSL_extension_supported.3ossl
share/man/man3/SSL_free.3ossl
share/man/man3/SSL_get0_connection.3ossl
share/man/man3/SSL_get0_group_name.3ossl
share/man/man3/SSL_get0_peer_rpk.3ossl
share/man/man3/SSL_get0_peer_scts.3ossl
share/man/man3/SSL_get_SSL_CTX.3ossl
share/man/man3/SSL_get_all_async_fds.3ossl
share/man/man3/SSL_get_certificate.3ossl
share/man/man3/SSL_get_ciphers.3ossl
share/man/man3/SSL_get_client_random.3ossl
share/man/man3/SSL_get_conn_close_info.3ossl
share/man/man3/SSL_get_current_cipher.3ossl
share/man/man3/SSL_get_default_timeout.3ossl
share/man/man3/SSL_get_error.3ossl
share/man/man3/SSL_get_event_timeout.3ossl
share/man/man3/SSL_get_extms_support.3ossl
share/man/man3/SSL_get_fd.3ossl
share/man/man3/SSL_get_handshake_rtt.3ossl
share/man/man3/SSL_get_peer_cert_chain.3ossl
share/man/man3/SSL_get_peer_certificate.3ossl
share/man/man3/SSL_get_peer_signature_nid.3ossl
share/man/man3/SSL_get_peer_tmp_key.3ossl
share/man/man3/SSL_get_psk_identity.3ossl
share/man/man3/SSL_get_rbio.3ossl
share/man/man3/SSL_get_rpoll_descriptor.3ossl
share/man/man3/SSL_get_session.3ossl
share/man/man3/SSL_get_shared_sigalgs.3ossl
share/man/man3/SSL_get_stream_id.3ossl
share/man/man3/SSL_get_stream_read_state.3ossl
share/man/man3/SSL_get_verify_result.3ossl
share/man/man3/SSL_get_version.3ossl
share/man/man3/SSL_group_to_name.3ossl
share/man/man3/SSL_handle_events.3ossl
share/man/man3/SSL_in_init.3ossl
share/man/man3/SSL_inject_net_dgram.3ossl
share/man/man3/SSL_key_update.3ossl
share/man/man3/SSL_library_init.3ossl
share/man/man3/SSL_load_client_CA_file.3ossl
share/man/man3/SSL_new.3ossl
share/man/man3/SSL_new_stream.3ossl
share/man/man3/SSL_pending.3ossl
share/man/man3/SSL_read.3ossl
share/man/man3/SSL_read_early_data.3ossl
share/man/man3/SSL_rstate_string.3ossl
share/man/man3/SSL_session_reused.3ossl
share/man/man3/SSL_set1_host.3ossl
share/man/man3/SSL_set1_initial_peer_addr.3ossl
share/man/man3/SSL_set1_server_cert_type.3ossl
share/man/man3/SSL_set_async_callback.3ossl
share/man/man3/SSL_set_bio.3ossl
share/man/man3/SSL_set_blocking_mode.3ossl
share/man/man3/SSL_set_connect_state.3ossl
share/man/man3/SSL_set_default_stream_mode.3ossl
share/man/man3/SSL_set_fd.3ossl
share/man/man3/SSL_set_incoming_stream_policy.3ossl
share/man/man3/SSL_set_retry_verify.3ossl
share/man/man3/SSL_set_session.3ossl
share/man/man3/SSL_set_shutdown.3ossl
share/man/man3/SSL_set_verify_result.3ossl
share/man/man3/SSL_shutdown.3ossl
share/man/man3/SSL_state_string.3ossl
share/man/man3/SSL_stream_conclude.3ossl
share/man/man3/SSL_stream_reset.3ossl
share/man/man3/SSL_want.3ossl
share/man/man3/SSL_write.3ossl
share/man/man3/TS_RESP_CTX_new.3ossl
share/man/man3/TS_VERIFY_CTX_set_certs.3ossl
share/man/man3/UI_STRING.3ossl
share/man/man3/UI_UTIL_read_pw.3ossl
share/man/man3/UI_create_method.3ossl
share/man/man3/UI_new.3ossl
share/man/man3/X509V3_get_d2i.3ossl
share/man/man3/X509V3_set_ctx.3ossl
share/man/man3/X509_ALGOR_dup.3ossl
share/man/man3/X509_CRL_get0_by_serial.3ossl
share/man/man3/X509_EXTENSION_set_object.3ossl
share/man/man3/X509_LOOKUP.3ossl
share/man/man3/X509_LOOKUP_hash_dir.3ossl
share/man/man3/X509_LOOKUP_meth_new.3ossl
share/man/man3/X509_NAME_ENTRY_get_object.3ossl
share/man/man3/X509_NAME_add_entry_by_txt.3ossl
share/man/man3/X509_NAME_get0_der.3ossl
share/man/man3/X509_NAME_get_index_by_NID.3ossl
share/man/man3/X509_NAME_print_ex.3ossl
share/man/man3/X509_PUBKEY_new.3ossl
share/man/man3/X509_REQ_get_extensions.3ossl
share/man/man3/X509_SIG_get0.3ossl
share/man/man3/X509_STORE_CTX_get_by_subject.3ossl
share/man/man3/X509_STORE_CTX_get_error.3ossl
share/man/man3/X509_STORE_CTX_new.3ossl
share/man/man3/X509_STORE_CTX_set_verify_cb.3ossl
share/man/man3/X509_STORE_add_cert.3ossl
share/man/man3/X509_STORE_get0_param.3ossl
share/man/man3/X509_STORE_new.3ossl
share/man/man3/X509_STORE_set_verify_cb_func.3ossl
share/man/man3/X509_VERIFY_PARAM_set_flags.3ossl
share/man/man3/X509_add_cert.3ossl
share/man/man3/X509_check_host.3ossl
share/man/man3/X509_check_issued.3ossl
share/man/man3/X509_check_private_key.3ossl
share/man/man3/X509_check_purpose.3ossl
share/man/man3/X509_cmp.3ossl
share/man/man3/X509_cmp_time.3ossl
share/man/man3/X509_digest.3ossl
share/man/man3/X509_dup.3ossl
share/man/man3/X509_get0_distinguishing_id.3ossl
share/man/man3/X509_get0_notBefore.3ossl
share/man/man3/X509_get0_signature.3ossl
share/man/man3/X509_get0_uids.3ossl
share/man/man3/X509_get_default_cert_file.3ossl
share/man/man3/X509_get_extension_flags.3ossl
share/man/man3/X509_get_pubkey.3ossl
share/man/man3/X509_get_serialNumber.3ossl
share/man/man3/X509_get_subject_name.3ossl
share/man/man3/X509_get_version.3ossl
share/man/man3/X509_load_http.3ossl
share/man/man3/X509_new.3ossl
share/man/man3/X509_sign.3ossl
share/man/man3/X509_verify.3ossl
share/man/man3/X509_verify_cert.3ossl
share/man/man3/X509v3_get_ext_by_NID.3ossl
share/man/man3/b2i_PVK_bio_ex.3ossl
share/man/man3/d2i_PKCS8PrivateKey_bio.3ossl
share/man/man3/d2i_PrivateKey.3ossl
share/man/man3/d2i_RSAPrivateKey.3ossl
share/man/man3/d2i_SSL_SESSION.3ossl
share/man/man3/d2i_X509.3ossl
share/man/man3/i2d_CMS_bio_stream.3ossl
share/man/man3/i2d_PKCS7_bio_stream.3ossl
share/man/man3/i2d_re_X509_tbs.3ossl
share/man/man3/o2i_SCT_LIST.3ossl
share/man/man3/s2i_ASN1_IA5STRING.3ossl
share/man/man5/config.5ossl
share/man/man5/fips_config.5ossl
share/man/man5/x509v3_config.5ossl
share/man/man7/EVP_ASYM_CIPHER-RSA.7ossl
share/man/man7/EVP_ASYM_CIPHER-SM2.7ossl
share/man/man7/EVP_CIPHER-AES.7ossl
share/man/man7/EVP_CIPHER-ARIA.7ossl
share/man/man7/EVP_CIPHER-BLOWFISH.7ossl
share/man/man7/EVP_CIPHER-CAMELLIA.7ossl
share/man/man7/EVP_CIPHER-CAST.7ossl
share/man/man7/EVP_CIPHER-CHACHA.7ossl
share/man/man7/EVP_CIPHER-DES.7ossl
share/man/man7/EVP_CIPHER-IDEA.7ossl
share/man/man7/EVP_CIPHER-NULL.7ossl
share/man/man7/EVP_CIPHER-RC2.7ossl
share/man/man7/EVP_CIPHER-RC4.7ossl
share/man/man7/EVP_CIPHER-RC5.7ossl
share/man/man7/EVP_CIPHER-SEED.7ossl
share/man/man7/EVP_CIPHER-SM4.7ossl
share/man/man7/EVP_KDF-ARGON2.7ossl
share/man/man7/EVP_KDF-HKDF.7ossl
share/man/man7/EVP_KDF-HMAC-DRBG.7ossl
share/man/man7/EVP_KDF-KB.7ossl
share/man/man7/EVP_KDF-KRB5KDF.7ossl
share/man/man7/EVP_KDF-PBKDF1.7ossl
share/man/man7/EVP_KDF-PBKDF2.7ossl
share/man/man7/EVP_KDF-PKCS12KDF.7ossl
share/man/man7/EVP_KDF-PVKKDF.7ossl
share/man/man7/EVP_KDF-SCRYPT.7ossl
share/man/man7/EVP_KDF-SS.7ossl
share/man/man7/EVP_KDF-SSHKDF.7ossl
share/man/man7/EVP_KDF-TLS13_KDF.7ossl
share/man/man7/EVP_KDF-TLS1_PRF.7ossl
share/man/man7/EVP_KDF-X942-ASN1.7ossl
share/man/man7/EVP_KDF-X942-CONCAT.7ossl
share/man/man7/EVP_KDF-X963.7ossl
share/man/man7/EVP_KEM-EC.7ossl
share/man/man7/EVP_KEM-RSA.7ossl
share/man/man7/EVP_KEM-X25519.7ossl
share/man/man7/EVP_KEYEXCH-DH.7ossl
share/man/man7/EVP_KEYEXCH-ECDH.7ossl
share/man/man7/EVP_KEYEXCH-X25519.7ossl
share/man/man7/EVP_MAC-BLAKE2.7ossl
share/man/man7/EVP_MAC-CMAC.7ossl
share/man/man7/EVP_MAC-GMAC.7ossl
share/man/man7/EVP_MAC-HMAC.7ossl
share/man/man7/EVP_MAC-KMAC.7ossl
share/man/man7/EVP_MAC-Poly1305.7ossl
share/man/man7/EVP_MAC-Siphash.7ossl
share/man/man7/EVP_MD-BLAKE2.7ossl
share/man/man7/EVP_MD-KECCAK.7ossl
share/man/man7/EVP_MD-MD2.7ossl
share/man/man7/EVP_MD-MD4.7ossl
share/man/man7/EVP_MD-MD5-SHA1.7ossl
share/man/man7/EVP_MD-MD5.7ossl
share/man/man7/EVP_MD-MDC2.7ossl
share/man/man7/EVP_MD-NULL.7ossl
share/man/man7/EVP_MD-RIPEMD160.7ossl
share/man/man7/EVP_MD-SHA1.7ossl
share/man/man7/EVP_MD-SHA2.7ossl
share/man/man7/EVP_MD-SHA3.7ossl
share/man/man7/EVP_MD-SHAKE.7ossl
share/man/man7/EVP_MD-SM3.7ossl
share/man/man7/EVP_MD-WHIRLPOOL.7ossl
share/man/man7/EVP_MD-common.7ossl
share/man/man7/EVP_PKEY-DH.7ossl
share/man/man7/EVP_PKEY-DSA.7ossl
share/man/man7/EVP_PKEY-EC.7ossl
share/man/man7/EVP_PKEY-FFC.7ossl
share/man/man7/EVP_PKEY-HMAC.7ossl
share/man/man7/EVP_PKEY-RSA.7ossl
share/man/man7/EVP_PKEY-SM2.7ossl
share/man/man7/EVP_PKEY-X25519.7ossl
share/man/man7/EVP_RAND-CTR-DRBG.7ossl
share/man/man7/EVP_RAND-HASH-DRBG.7ossl
share/man/man7/EVP_RAND-HMAC-DRBG.7ossl
share/man/man7/EVP_RAND-SEED-SRC.7ossl
share/man/man7/EVP_RAND-TEST-RAND.7ossl
share/man/man7/EVP_RAND.7ossl
share/man/man7/EVP_SIGNATURE-DSA.7ossl
share/man/man7/EVP_SIGNATURE-ECDSA.7ossl
share/man/man7/EVP_SIGNATURE-ED25519.7ossl
share/man/man7/EVP_SIGNATURE-HMAC.7ossl
share/man/man7/EVP_SIGNATURE-RSA.7ossl
share/man/man7/OSSL_PROVIDER-FIPS.7ossl
share/man/man7/OSSL_PROVIDER-base.7ossl
share/man/man7/OSSL_PROVIDER-default.7ossl
share/man/man7/OSSL_PROVIDER-legacy.7ossl
share/man/man7/OSSL_PROVIDER-null.7ossl
share/man/man7/RAND.7ossl
share/man/man7/RSA-PSS.7ossl
share/man/man7/X25519.7ossl
share/man/man7/bio.7ossl
share/man/man7/ct.7ossl
share/man/man7/des_modes.7ossl
share/man/man7/evp.7ossl
share/man/man7/fips_module.7ossl
share/man/man7/life_cycle-cipher.7ossl
share/man/man7/life_cycle-digest.7ossl
share/man/man7/life_cycle-kdf.7ossl
share/man/man7/life_cycle-mac.7ossl
share/man/man7/life_cycle-pkey.7ossl
share/man/man7/life_cycle-rand.7ossl
share/man/man7/openssl-core.h.7ossl
share/man/man7/openssl-core_dispatch.h.7ossl
share/man/man7/openssl-core_names.h.7ossl
share/man/man7/openssl-env.7ossl
share/man/man7/openssl-glossary.7ossl
share/man/man7/openssl-quic.7ossl
share/man/man7/openssl-threads.7ossl
share/man/man7/openssl_user_macros.7ossl
share/man/man7/ossl-guide-introduction.7ossl
share/man/man7/ossl-guide-libcrypto-introduction.7ossl
share/man/man7/ossl-guide-libraries-introduction.7ossl
share/man/man7/ossl-guide-libssl-introduction.7ossl
share/man/man7/ossl-guide-migration.7ossl
share/man/man7/ossl-guide-quic-client-block.7ossl
share/man/man7/ossl-guide-quic-client-non-block.7ossl
share/man/man7/ossl-guide-quic-introduction.7ossl
share/man/man7/ossl-guide-quic-multi-stream.7ossl
share/man/man7/ossl-guide-tls-client-block.7ossl
share/man/man7/ossl-guide-tls-client-non-block.7ossl
share/man/man7/ossl-guide-tls-introduction.7ossl
share/man/man7/ossl_store-file.7ossl
share/man/man7/ossl_store.7ossl
share/man/man7/passphrase-encoding.7ossl
share/man/man7/property.7ossl
share/man/man7/provider-asym_cipher.7ossl
share/man/man7/provider-base.7ossl
share/man/man7/provider-cipher.7ossl
share/man/man7/provider-decoder.7ossl
share/man/man7/provider-digest.7ossl
share/man/man7/provider-encoder.7ossl
share/man/man7/provider-kdf.7ossl
share/man/man7/provider-kem.7ossl
share/man/man7/provider-keyexch.7ossl
share/man/man7/provider-keymgmt.7ossl
share/man/man7/provider-mac.7ossl
share/man/man7/provider-object.7ossl
share/man/man7/provider-rand.7ossl
share/man/man7/provider-signature.7ossl
share/man/man7/provider-storemgmt.7ossl
share/man/man7/provider.7ossl
share/man/man7/proxy-certificates.7ossl
share/man/man7/x509.7ossl
share/doc/openssl
EOM
}



