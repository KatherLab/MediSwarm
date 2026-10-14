#!/usr/bin/env python3
"""Print the image id a site's heartbeat reports for a pushed image, and who runs it.

    scripts/build/heartbeat_image_id.py jefftud/odelia:1.9.0 [more refs ...]
    scripts/build/heartbeat_image_id.py --live-root /srv/mediswarm/live jefftud/odelia:1.9.0

build_heartbeat.sh records `docker inspect --format '{{.Image}}' <client container>` cut to 19
characters. On the sites' Docker (classic image store) that is the digest of the image CONFIG
for linux/amd64 -- neither the tag's manifest digest shown on Docker Hub nor the id that
`docker images` prints on a host with the containerd image store (agh1). This resolves the
registry's index to the amd64 manifest and prints its config digest, short form first.

With --live-root it also reads each site's newest heartbeat.json under
<live-root>/<SITE>/swarm/<job>/ and marks whether that site runs the FIRST image given.
"""

import argparse
import glob
import json
import os
import subprocess
import sys


def raw(ref):
    out = subprocess.run(["docker", "buildx", "imagetools", "inspect", "--raw", ref],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def repository(ref):
    """'jefftud/odelia:1.9.0' -> 'jefftud/odelia'; a registry port (host:5000/x) is not a tag."""
    name = ref.split("@", 1)[0]
    head, sep, tail = name.rpartition(":")
    return head if sep and "/" not in tail else name


def config_digest(ref, arch="amd64", os_name="linux"):
    doc = raw(ref)
    if "manifests" in doc:  # an index: pick the platform manifest, skip attestations
        picks = [m["digest"] for m in doc["manifests"]
                 if m.get("platform", {}).get("architecture") == arch
                 and m.get("platform", {}).get("os") == os_name]
        if len(picks) != 1:
            raise SystemExit(f"{ref}: expected one {os_name}/{arch} manifest, found {len(picks)}")
        doc = raw(f"{repository(ref)}@{picks[0]}")
    return doc["config"]["digest"]


def newest_heartbeats(live_root):
    """{site: heartbeat dict} from each site's most recently written swarm heartbeat."""
    found = {}
    for site_dir in sorted(glob.glob(os.path.join(live_root, "*"))):
        beats = glob.glob(os.path.join(site_dir, "swarm", "*", "heartbeat.json"))
        if not beats:
            continue
        newest = max(beats, key=os.path.getmtime)
        try:
            with open(newest) as fh:
                found[os.path.basename(site_dir)] = json.load(fh)
        except (OSError, ValueError):
            found[os.path.basename(site_dir)] = {}
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+", help="image references, e.g. jefftud/odelia:1.9.0")
    ap.add_argument("--live-root", help="live monitor root holding <SITE>/swarm/<job>/heartbeat.json")
    args = ap.parse_args(argv)

    expected = None
    for ref in args.refs:
        digest = config_digest(ref)
        expected = expected or digest[:19]
        print(f"{digest[:19]}  {digest}  {ref}")

    if args.live_root:
        print(f"\nsites (newest heartbeat) against {args.refs[0]} = {expected}:")
        for site, beat in newest_heartbeats(args.live_root).items():
            image_id = beat.get("image_id") or "-"
            mark = "OK " if image_id == expected else "OLD"
            print(f"  {mark} {site:<10} {image_id:<20} {beat.get('timestamp', '?')}  kit {beat.get('kit_version', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
