#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalize this copied package's Python files after Windows-to-Linux transfer.

Run on the Linux validation copy before catkin_make, not on a mounted source
tree that must retain original bytes. Keep the transferred archive as backup.
"""
from __future__ import print_function
import os
import stat


def main():
    if os.name != 'posix':
        raise RuntimeError('Run in the copied Linux package, not on Windows')
    package = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isfile(os.path.join(package, 'package.xml')):
        raise RuntimeError('Package root not found')
    changed = 0
    for directory, subdirs, files in os.walk(package):
        subdirs[:] = [name for name in subdirs if name not in ('.git', '__pycache__')]
        for name in files:
            if not name.endswith('.py'):
                continue
            path = os.path.join(directory, name)
            if os.path.islink(path):
                continue
            with open(path, 'rb') as stream:
                original = stream.read()
            normalized = original.replace(b'\r\n', b'\n')
            if original != normalized:
                with open(path, 'wb') as stream:
                    stream.write(normalized)
                changed += 1
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print('Package: '+package)
    print('Normalized Python files: %d; Python executable permissions set' % changed)


if __name__ == '__main__':
    main()
