#!/usr/bin/env python3

import re
import sys

LINE_BREAK_IN_COMMAND = ' \\\n    '
LINE_BREAK_REPLACEMENT = ' λινε βρεακ ρεπλαζεμεντ '

def load_file(filename: str) -> str:
    with open(filename, 'r') as infile:
        return infile.read()

def save_file(contents: str, filename: str) -> None:
    with open(filename, 'w') as outfile:
        outfile.write(contents)


def remove_apt_versions(contents: str) -> str:
    contents = contents.replace(LINE_BREAK_IN_COMMAND, LINE_BREAK_REPLACEMENT)
    output = []
    for line in contents.splitlines():
        if line.startswith('RUN apt install -y'):
            out_line = re.sub('=[^ ]*', '', line)
            output.append(out_line)
        else:
            output.append(line)
    output = '\n'.join(output) + '\n'
    output = output.replace(LINE_BREAK_REPLACEMENT, LINE_BREAK_IN_COMMAND)
    return output

if __name__ == '__main__':
    contents = load_file(sys.argv[1])
    if LINE_BREAK_REPLACEMENT in contents:
        raise Exception('Line break replacement {LINE_BREAK_REPLACEMENT} in Dockerfile, cannot process it.')
    contents = remove_apt_versions(contents)
    save_file(contents, sys.argv[1])
