#!/usr/bin/env bash

# Roughly following https://www.digitalocean.com/community/tutorials/how-to-set-up-and-configure-an-openvpn-server-on-ubuntu-20-04
# but on 22.04

chown ca_user:ca_user /home/ca_user/ -R
chmod a+rwX /home/ca_user/ -R
/bin/su - -c '/home/ca_user/ca_setup.sh' ca_user

mkdir ~/easy-rsa
ln -s /usr/share/easy-rsa/* ~/easy-rsa/
cd ~/easy-rsa

echo 'set_var EASYRSA_ALGO   "ec"'     >  vars
echo 'set_var EASYRSA_DIGEST "sha512"' >> vars

./easyrsa init-pki

rm /server_config/ca.crt \
   /server_config/server.crt \
   /server_config/server.key \
   /server_config/ta.key -f

rm -rf /client_configs/keys
mkdir -p /client_configs/keys/

export EASYRSA_BATCH=1
./easyrsa gen-req server nopass

cp ~/easy-rsa/pki/reqs/server.req /tmp/
chmod a+r /tmp/server.req
/bin/su - -c "export EASYRSA_BATCH=1 && cd ~/easy-rsa/ && ./easyrsa import-req /tmp/server.req server && ./easyrsa sign-req server server" ca_user

cd ~/easy-rsa
openvpn --genkey secret ta.key
cp ta.key /client_configs/keys/
cp /home/ca_user/easy-rsa/pki/ca.crt /client_configs/keys/

# copy/create files to where they are needed
cp /home/ca_user/easy-rsa/pki/ca.crt             /server_config/
cp /home/ca_user/easy-rsa/pki/issued/server.crt  /server_config/
cp ~/easy-rsa/pki/private/server.key             /server_config/
cp ~/easy-rsa/ta.key                             /server_config/

mkdir /server_config/ccd

i=4
for client in testserver.local admin@test.odelia client_A client_B; do
    cd ~/easy-rsa
    EASYRSA_BATCH=1 EASYRSA_REQ_CN=$client ./easyrsa gen-req $client nopass
    cp pki/private/$client.key /client_configs/keys/

    cp ~/easy-rsa/pki/reqs/$client.req /tmp/
    chmod a+r /tmp/$client.req
    /bin/su - -c "export EASYRSA_BATCH=1 && cd ~/easy-rsa/ && ./easyrsa import-req /tmp/$client.req $client && ./easyrsa sign-req client $client" ca_user
    cp /home/ca_user/easy-rsa/pki/issued/$client.crt /client_configs/keys/

    cd /client_configs
    ./make_ovpn.sh $client

    echo "ifconfig-push 10.8.0."$i" 255.0.0.0" > /server_config/ccd/$client
    i=$((i+1))
done

chmod a+rwX /client_configs -R
chmod a+rwX /server_config -R
chmod a+rwX /home/ca_user -R
