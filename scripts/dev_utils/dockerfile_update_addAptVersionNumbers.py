#!/usr/bin/env python3

import re
import sys

def load_file(filename: str) -> str:
    with open(filename, 'r') as infile:
        return infile.read()

def save_file(contents: str, filename: str) -> None:
    with open(filename, 'w') as outfile:
        outfile.write(contents)


def parse_apt_versions(installlog: str) -> dict:
    versions = {}
    pattern = re.compile(r'Get:.*? ([a-z0-9\-\+\.]+)(?:/[^ ]*)? ([0-9a-zA-Z\:\~\.\+\-]+) ')
    for line in installlog.splitlines():
        match = pattern.search(line)
        if match:
            package = match.group(1)
            version = match.group(2)
            if package in versions and versions[package] != version:
                print(f'Conflicting versions of {package} found: {versions[package]} and {version} found, using the latter.')
            versions[package] = version
    return versions



def add_apt_versions(dockerfile: str, versions: dict) -> str:
    dockerfile = dockerfile.replace('RUN apt install', 'RUN_apt_install')
    outlines = []
    for line in dockerfile.splitlines():
        if line.startswith('RUN_apt_install'):
            outline = '' + line
            for package, version in versions.items():
                outline = outline.replace(f' {package} ', f' {package}={version} ')
                outline = re.sub(f' {package}$', f' {package}={version}', outline)
            parts = outline.split()
            if len(parts) > 3:
                header = " ".join(parts[:3])
                pkgs = parts[3:]
                outline = header + " \\\n    " + " \\\n    ".join(pkgs)
            outlines.append(outline)
        else:
            outlines.append(line)
    dockerfile = '\n'.join(outlines) + '\n'
    dockerfile = dockerfile.replace('RUN_apt_install', 'RUN apt install')
    return dockerfile


def report_non_fixed_versions(dockerfile: str, versions: dict) -> None:
    for package in versions.keys():
        if package not in dockerfile:
            print(f'Package {package} does not have a fixed version')


if __name__ == '__main__':
    dockerfile = load_file(sys.argv[1])
    installlog = load_file(sys.argv[2])
    versions = parse_apt_versions(installlog)
    report_non_fixed_versions(dockerfile, versions)
    dockerfile = add_apt_versions(dockerfile, versions)
    save_file(dockerfile, sys.argv[1])