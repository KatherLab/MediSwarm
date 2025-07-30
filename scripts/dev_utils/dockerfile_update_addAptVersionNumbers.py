#!/usr/bin/env python3

import re
import sys

from dockerfile_update_removeVersionApt import LINE_BREAK_IN_COMMAND, LINE_BREAK_REPLACEMENT, load_file, save_file

APT_INSTALL_COMMAND = 'RUN apt install -y'
APT_INSTALL_REPLACEMENT = 'ΡΥΝ απτ ινσταλλ -υ'

def parse_apt_versions(installlog: str) -> dict:
    versions = {}
    for line in installlog.splitlines():
        if re.match('.*Get:[0-9]* http.*', line):
            blocks = line.split(' ')
            if len(blocks) > 9:
                package = blocks[6]
                version = blocks[8]
                if package in versions and versions[package] != version:
                    print(f'Conflicting versions of {package} found: {versions[package]} and {version} found, using the latter.')
                versions[package] = version
    return versions


def add_apt_versions(dockerfile: str, versions: dict) -> str:
    dockerfile = dockerfile.replace(LINE_BREAK_IN_COMMAND, LINE_BREAK_REPLACEMENT)
    dockerfile = dockerfile.replace(APT_INSTALL_COMMAND, APT_INSTALL_REPLACEMENT)
    outlines = []
    for line in dockerfile.splitlines():
        if line.startswith(APT_INSTALL_REPLACEMENT):
            outline = '' + line
            for package, version in versions.items():
                outline = outline.replace(f' {package} ', f' {package}={version} ')
                outline = re.sub(f' {package}$', f' {package}={version}', outline)
            outlines.append(outline)
        else:
            outlines.append(line)
    dockerfile = '\n'.join(outlines) + '\n'
    dockerfile = dockerfile.replace(APT_INSTALL_REPLACEMENT, APT_INSTALL_COMMAND)
    dockerfile = dockerfile.replace(LINE_BREAK_REPLACEMENT, LINE_BREAK_IN_COMMAND)
    return dockerfile


def report_non_fixed_versions(dockerfile: str, versions: dict) -> None:
    for package in versions.keys():
        if package not in dockerfile:
            print(f'Package {package} does not have a fixed version')


if __name__ == '__main__':
    dockerfile = load_file(sys.argv[1])
    installlog = load_file(sys.argv[2])
    if LINE_BREAK_REPLACEMENT in dockerfile or APT_INSTALL_REPLACEMENT in dockerfile:
        raise Exception('Line break replacement {LINE_BREAK_REPLACEMENT} or apt command replacement {APT_INSTALL_REPLACEMENT} in Dockerfile, cannot process it.')

    versions = parse_apt_versions(installlog)
    report_non_fixed_versions(dockerfile, versions)
    dockerfile = add_apt_versions(dockerfile, versions)
    save_file(dockerfile, sys.argv[1])
