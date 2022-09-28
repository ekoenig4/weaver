import os
import subprocess
import shutil
from glob import glob as local_glob


class eos:
    url = "root://cmseos.fnal.gov/"

    @staticmethod
    def ls_with_uscms(path, with_path):
        cmd = ['ls', '/eos/uscms'+path]
        stdout = subprocess.run(
            [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
        dirlist = stdout.strip().split('\n')

        if with_path:
            return [ d.replace('/eos/uscms','') for d in dirlist ]
        else:
            return [ os.path.basename(d) for d in dirlist ]
        
    @staticmethod
    def ls_with_eos(path, with_path=False):
        cmd = ['eos', eos.url, 'ls', path]
        stdout = subprocess.run(
            [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
        dirlist = stdout.strip().split('\n')

        if with_path:
            path = os.path.dirname(path)
            return [f'{path}/{d}' for d in dirlist]
        return dirlist

    @staticmethod
    def ls(path, with_path=False, with_eos=False):
        if with_eos: 
            dirlist = eos.ls_with_eos(path, with_path)
        else:
            dirlist = eos.ls_with_uscms(path, with_path)

        return dirlist

    @staticmethod
    def exists(path):
        cmd = ['eos', eos.url, 'ls', path]
        stdout = subprocess.run(
            [' '.join(cmd)], shell=True, capture_output=True).stdout.decode("utf-8")
        stdout.strip()
        return any(stdout)

    @staticmethod
    def copy(src, dest,):
        cmd = ['xrdcp','-f', src, dest]
        return subprocess.run(
            [' '.join(cmd)], shell=True)

    @staticmethod
    def fullpath(path):
        return eos.url + path

def glob(path):
    filelist = local_glob(path)
    if any(filelist):
        return filelist
    filelist = eos.ls(path, with_path=True)
    if any(filelist):
        return filelist
    return []

def copy(src, dest):
    if os.path.exists(src):
        return shutil.copy2(src, dest)
    if eos.exists(src):
        return copy_from_eos(src, dest)

def copy_to_eos(src, dest):
    return eos.copy(src, eos.fullpath(dest))

def copy_from_eos(src, dest):
    return eos.copy(eos.fullpath(src), dest)

def fullpath(path):
    if os.path.exists(path):
        return os.path.abspath(path)
    if eos.exists(path):
        return eos.fullpath(path)
    return path

def remove_eos(path):
    return path.replace('/eos/uscms', '')
