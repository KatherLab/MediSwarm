#!/usr/bin/env python3

import re
import sys

def load_file(filename: str) -> str:
    with open(filename, 'r') as infile:
        return infile.read()

def save_file(contents: str, filename: str) -> None:
    with open(filename, 'w') as outfile:
        outfile.write(contents)


def remove_apt_versions(dockerfile: str) -> str:
    output = []
    for line in dockerfile.splitlines():
        if line.startswith('RUN apt install'):
            out_line = re.sub('=[^ ]*', '', line)
            parts = out_line.split()
            if len(parts) > 3:
                header = " ".join(parts[:3])
                pkgs = parts[3:]
                out_line = header + " \\\n    " + " \\\n    ".join(pkgs)
            output.append(out_line)
        else:
            output.append(line)
    return '\n'.join(output)


if __name__ == '__main__':
    dockerfile = load_file(sys.argv[1])
    dockerfile = remove_apt_versions(dockerfile)
    save_file(dockerfile, sys.argv[1])