#!/usr/bin/env bash

rm -rf ~/easy-rsa
mkdir ~/easy-rsa
ln -s /usr/share/easy-rsa/* ~/easy-rsa/
cd ~/easy-rsa
./easyrsa init-pki

echo 'set_var EASYRSA_REQ_COUNTRY    "DE"'                 >  vars
echo 'set_var EASYRSA_REQ_PROVINCE   "Bremen"'             >> vars
echo 'set_var EASYRSA_REQ_CITY       "Bremen"'             >> vars
echo 'set_var EASYRSA_REQ_ORG        "ODELIA_MEVIS"'       >> vars
echo 'set_var EASYRSA_REQ_EMAIL      "admin@mevis.odelia"' >> vars
echo 'set_var EASYRSA_REQ_OU         "Testing"'            >> vars
echo 'set_var EASYRSA_ALGO           "ec"'                 >> vars
echo 'set_var EASYRSA_DIGEST         "sha512"'             >> vars

export EASYRSA_BATCH=1
./easyrsa build-ca nopass
