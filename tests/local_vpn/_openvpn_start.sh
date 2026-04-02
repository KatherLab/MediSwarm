#!/usr/bin/env bash

echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
sysctl -p

echo "MTBhMTEsMTkKPiAjIFNUQVJUIE9QRU5WUE4gUlVMRVMKPiAjIE5BVCB0YWJsZSBydWxlcwo+ICpuYXQKPiA6UE9TVFJPVVRJTkcgQUNDRVBUIFswOjBdCj4gIyBBbGxvdyB0cmFmZmljIGZyb20gT3BlblZQTiBjbGllbnQgdG8gZXRoMCAoY2hhbmdlIHRvIHRoZSBpbnRlcmZhY2UgeW91IGRpc2NvdmVyZWQhKQo+IC1BIFBPU1RST1VUSU5HIC1zIDEwLjguMC4wLzggLW8gZXRoMCAtaiBNQVNRVUVSQURFCj4gQ09NTUlUCj4gIyBFTkQgT1BFTlZQTiBSVUxFUwo+IAo=" | base64 -d > before.rules.patch
patch /etc/ufw/before.rules before.rules.patch
rm before.rules.patch

echo "MTljMTkKPCBERUZBVUxUX0ZPUldBUkRfUE9MSUNZPSJEUk9QIgotLS0KPiBERUZBVUxUX0ZPUldBUkRfUE9MSUNZPSJBQ0NFUFQiCg==" | base64 -d > ufw.patch
patch /etc/default/ufw ufw.patch
rm ufw.patch

ufw allow 9194/udp
ufw allow OpenSSH
ufw disable
ufw enable

cp /server_config/ca.crt      /etc/openvpn/server/
cp /server_config/server.conf /etc/openvpn/server/
cp /server_config/server.crt  /etc/openvpn/server/
cp /server_config/server.key  /etc/openvpn/server/
cp /server_config/ta.key      /etc/openvpn/server/
cp /server_config/ccd         /etc/openvpn/ccd     -r

# write log to folder on host
cd server_config

nohup openvpn --duplicate-cn --client-to-client --config /etc/openvpn/server/server.conf &
sleep 2
chmod a+r /server_config/nohup.out

tc qdisc add dev eth0 root tbf rate 60mbit burst 5mbit limit 16gbit
