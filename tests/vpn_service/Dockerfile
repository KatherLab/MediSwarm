# Test image for the offline mediswarm-vpn reproduction (tests/vpn_service/run_vpn_service_repro.sh).
#
# One image serves both roles: the OpenVPN test gateway (harness/server_entry.sh) and the site host, where
# setup_vpntunnel.sh runs as root against a small systemctl/journalctl stand-in (harness/fake_systemd.sh).
# Real systemd would need CAP_SYS_ADMIN; both containers only get NET_ADMIN and /dev/net/tun.
#
# Built online once; the reproduction itself runs on an --internal docker network without internet.
ARG UBUNTU=22.04
FROM ubuntu:${UBUNTU}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        openvpn openssl iproute2 iputils-ping procps sudo util-linux expect ca-certificates hostname \
 && rm -rf /var/lib/apt/lists/*

# setup_vpntunnel.sh -n runs "sudo apt-get update/install openvpn". The reproduction runs offline and
# openvpn is installed already, so a stand-in earlier in sudo's secure_path answers instead.
RUN printf '#!/bin/sh\necho "[repro] apt-get $* (skipped: offline test, openvpn preinstalled)"\nexit 0\n' \
        > /usr/local/sbin/apt-get \
 && chmod +x /usr/local/sbin/apt-get

COPY harness/fake_systemd.sh harness/drive_setup.exp /usr/local/lib/fake-systemd/
RUN chmod 755 /usr/local/lib/fake-systemd/fake_systemd.sh /usr/local/lib/fake-systemd/drive_setup.exp \
 && ln -s /usr/local/lib/fake-systemd/fake_systemd.sh /usr/local/sbin/systemctl \
 && ln -s /usr/local/lib/fake-systemd/fake_systemd.sh /usr/local/sbin/journalctl \
 && mkdir -p /etc/systemd/system

# The site user whose NetworkManager export keeps its certificates under ~/.cert/nm-openvpn/.
RUN useradd -m siteuser && chmod 700 /home/siteuser
