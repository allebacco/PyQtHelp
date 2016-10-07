#!/usr/bin/env python

import os
import sys
import shlex
import fnmatch
import subprocess
from collections import namedtuple
try:
    from ConfigParser import SafeConfigParser
except ImportError:
    from configparser import SafeConfigParser

from setuptools import setup
from distutils.core import Extension
from distutils import dir_util, spawn, log

import glob

import sipdistutils
from PyQt5.QtCore import PYQT_CONFIGURATION

from test_command import TestCommand

pjoin = os.path.join

NAME = 'PyQtHelp'
DESCRIPTION = 'PyQt native functions'
DESCRIPTION_TAGS = 'pyqt pnative functions'
URL = "http://example.com"
LICENSE = 'MIT'
AUTHOR = "Alessandro Bacchini"
AUTHOR_EMAIL = "allebacco@gmail.com"
VERSION = '0.0.0'


pyqt_conf = namedtuple("pyqt_conf", ["sip_flags", "sip_dir"])
qt_conf = namedtuple("qt_conf", ["prefix", "include_dir", "library_dir", "framework", "framework_dir"])
config = namedtuple("config", ["sip", "pyqt_conf", "qt_conf"])

pyqt_sip_dir = None
pyqt_sip_flags = PYQT_CONFIGURATION["sip_flags"]

qt_dir = None
qt_include_dir = None
qt_lib_dir = None
qt_bin_dir = None
qt_framework = False
qt_framework_dir = None
qt_moc = 'moc'
qt_qmake = 'qmake'

if 'MOC' in os.environ:
    qt_moc = os.environ['MOC']
if 'QMAKE' in os.environ:
    qt_qmake = os.environ['QMAKE']

# if QTDIR env is defined use it
if "QTDIR" in os.environ and len(os.environ["QTDIR"]) > 0:
    qt_dir = os.environ["QTDIR"]
    if sys.platform == "darwin":
        if glob.glob(pjoin(qt_dir, "lib", "Qt*.framework")):
            # This is the standard Qt framework layout
            qt_framework = True
            qt_framework_dir = pjoin(qt_dir, "lib")
        elif glob.glob(pjoin(qt_dir, "Frameworks", "Qt*.framework")):
            # Also worth checking (standard for bundled apps)
            qt_framework = True
            qt_framework_dir = pjoin(qt_dir, "Frameworks")

    if not qt_framework:
        # Assume standard layout
        qt_framework = False
        qt_include_dir = pjoin(qt_dir, "include")
        qt_lib_dir = pjoin(qt_dir, "lib")
        qt_framework_dir = qt_dir

    qt_bin_dir = pjoin(qt_dir, 'bin')


if qt_framework is False:
    # Detect failed: try to force autodetect
    print(sys.version.lower())
    if 'anaconda' in sys.version.lower():
        # We are in the Continuum Analytics' Anaconda environment
        # and it is possible to autodetect Qt5 configuration
        print('Detected Anaconda environment')
        qt_conf_filename = pjoin(sys.prefix, 'qt.conf')
        conf_parser = SafeConfigParser()
        conf_parser.read([qt_conf_filename, 'qt.conf'])
        qt_dir = conf_parser.get('Paths', 'Prefix')
        qt_bin_dir = conf_parser.get('Paths', 'Binaries')
        qt_lib_dir = conf_parser.get('Paths', 'Libraries')
        qt_include_dir = conf_parser.get('Paths', 'Headers')
        qt_framework_dir = qt_bin_dir
        if qt_moc is None:
            qt_moc = pjoin(qt_bin_dir, 'moc')
        if qt_qmake is None:
            qt_qmake = pjoin(qt_bin_dir, 'qmake')
    elif 'posix' in os.name:
        # We are in the Linux environment
        # and it is possible to autodetect Qt5 configuration
        print('Detected Linux environment')
        output = subprocess.getoutput(qt_qmake + ' -query')
        output = output.split('\n')
        output = {s[0]: s[1] for s in (o.split(':') for o in output)}

        qt_dir = output['QT_INSTALL_PREFIX']
        qt_include_dir = output['QT_INSTALL_HEADERS']
        qt_lib_dir = output['QT_INSTALL_LIBS']
        qt_bin_dir = output['QT_INSTALL_BINS']
        #qt_framework = True
        qt_framework_dir = output['QT_INSTALL_PREFIX']
        qt_moc = os.path.join(qt_bin_dir, qt_moc)
        qt_qmake = os.path.join(qt_bin_dir, qt_qmake)


def which(name):
    """
    Return the path of program named 'name' on the $PATH.
    """
    if os.name == "nt" and not name.endswith(".exe"):
        name = name + ".exe"

    for path in os.environ["PATH"].split(os.pathsep):
        path = os.path.join(path, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None

qt_moc = which(qt_moc)
if qt_moc is None:
    raise RuntimeError('Unable to locate Qt moc')
qt_qmake = which(qt_qmake)
if qt_qmake is None:
    raise RuntimeError('Unable to locate Qt qmake')

src_dir = pjoin(os.path.dirname(os.path.abspath(__file__)), 'src')

extra_compile_args = []
extra_link_args = []
include_dirs = []
library_dirs = []


def site_config():
    parser = SafeConfigParser(dict(os.environ))
    parser.read(["site.cfg"])

    def get(section, option, default=None, type=None):
        if parser.has_option(section, option):
            if type is None:
                return parser.get(section, option)
            elif type is bool:
                return parser.getboolean(section, option)
            elif type is int:
                return parser.getint(section, option)
            else:
                raise TypeError
        else:
            return default

    sip_bin = get("sip", "sip_bin")

    sip_flags = get("pyqt", "sip_flags", default=pyqt_sip_flags)
    sip_dir = get("pyqt", "sip_dir", default=pyqt_sip_dir)

    if sip_flags is not None:
        sip_flags = shlex.split(sip_flags)
    else:
        sip_flags = []

    prefix = get("qt", "qt_dir", default=qt_dir)
    include_dir = get("qt", "include_dir", default=qt_include_dir)
    library_dir = get("qt", "library_dir", default=qt_lib_dir)
    framework = get("qt", "framework", default=qt_framework, type=bool)
    framework_dir = get("qt", "framework_dir", default=qt_framework_dir)

    def path_list(path):
        if path and path.strip():
            return path.split(os.pathsep)
        else:
            return []

    include_dir = path_list(include_dir)
    library_dir = path_list(library_dir)

    for d in ['QtCore', 'QtGui']:
        include_dir.append(pjoin(qt_include_dir, d))

    conf = config(sip_bin, pyqt_conf(sip_flags, sip_dir),
                  qt_conf(prefix, include_dir, library_dir, framework, framework_dir))
    return conf

site_cfg = site_config()
print('site_cfg', site_cfg)

if sys.platform == "darwin":
    sip_plaftorm_tag = "WS_MACX"
elif sys.platform == "win32":
    sip_plaftorm_tag = "WS_WIN"
elif sys.platform.startswith("linux"):
    sip_plaftorm_tag = "WS_X11"
else:
    sip_plaftorm_tag = ""


class PyQt5Extension(Extension):
    pass


class build_pyqt_ext(sipdistutils.build_ext):
    """
    A build_ext command for building PyQt5 sip based extensions
    """
    description = "Build a orangeqt PyQ5 extension."

    def finalize_options(self):
        sipdistutils.build_ext.finalize_options(self)
        self.sip_opts = self.sip_opts + site_cfg.pyqt_conf.sip_flags
        self.sip_opts += ['-I./src ', '-e']

        import numpy
        self.sip_opts += ['-I%s ' % numpy.get_include()]

    def build_extension(self, ext):
        if not isinstance(ext, PyQt5Extension):
            return

        cppsources = [source for source in ext.sources if source.endswith(".cpp")]

        dir_util.mkpath(self.build_temp, dry_run=self.dry_run)

        # Run moc on all header files.
        for source in cppsources:
            header = source.replace(".cpp", ".h")
            if os.path.exists(header):
                moc_file = os.path.basename(header).replace(".h", ".moc")
                out_file = os.path.join(self.build_temp, moc_file)
                call_arg = [qt_moc, "-o", out_file, header]
                spawn.spawn(call_arg, dry_run=self.dry_run)

        import numpy
        # Add the temp build directory to include path, for compiler to find
        # the created .moc files
        ext.include_dirs = ext.include_dirs + [numpy.get_include(), self.build_temp]

        sipdistutils.build_ext.build_extension(self, ext)

    def _find_sip(self):
        if site_cfg.sip:
            log.info("Using sip at %r (from .cfg file)" % site_cfg.sip)
            return site_cfg.sip

        # Try the base implementation
        sip = sipdistutils.build_ext._find_sip(self)
        if os.path.isfile(sip):
            return sip

        log.warn("Could not find sip executable at %r." % sip)

        # Find sip on $PATH
        sip = which("sip")

        if sip:
            log.info("Found sip on $PATH at: %s" % sip)
            return sip

        return sip

    # For sipdistutils to find PyQt4's .sip files
    def _sip_sipfiles_dir(self):
        if site_cfg.pyqt_conf.sip_dir:
            return site_cfg.pyqt_conf.sip_dir

        if pyqt_sip_dir is not None and os.path.isdir(pyqt_sip_dir):
            return pyqt_sip_dir

        log.warn("The default sip include directory %r does not exist" % pyqt_sip_dir)

        candidate_sipfiles_dirs = [pjoin(sys.prefix, "share/sip/PyQt5"),
                                   pjoin(sys.prefix, "sip/PyQt5"),]
        for path in candidate_sipfiles_dirs:
            if os.path.isdir(path):
                log.info("Found sip include directory at %r" % path)
                return path

        return "."


def get_source_files(path, ext="cpp", exclude=tuple()):
    out_files = list()
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, ext):
                filename = os.path.join(root, basename)
                out_files.append(filename)

    out_files = [f for f in out_files if os.path.basename(f) not in exclude]
    return out_files


# Used Qt5 libraries
qt_libs = ["Qt5Core", "Qt5Gui"]

if site_cfg.qt_conf.framework:
    framework_dir = site_cfg.qt_conf.framework_dir
    extra_compile_args = ["-F%s" % framework_dir]
    extra_link_args = ["-F%s" % framework_dir]
    for lib in qt_libs:
        include_dirs += [os.path.join(framework_dir,
                                      lib + ".framework", "Headers")]
        extra_link_args += ["-framework", lib]
else:
    if type(site_cfg.qt_conf.include_dir) == list:
        include_dirs = site_cfg.qt_conf.include_dir + [pjoin(d, lib) for lib in qt_libs for d in site_cfg.qt_conf.include_dir]
    else:
        include_dirs = [site_cfg.qt_conf.include_dir] + \
                       [pjoin(site_cfg.qt_conf.include_dir, lib) for lib in qt_libs]
    library_dirs += site_cfg.qt_conf.library_dir


include_dirs += ["./", "./src"]
source_files = get_source_files("./src", "*.cpp")
print('Cpp source files:', source_files)

if os.name == "nt":
    extra_compile_args += ['/fp:precise']
else:
    extra_compile_args += ['-std=c++11']

print('source_files', source_files)

pyqthelp_ext = PyQt5Extension("pyqthelp.native",
                              ["pyqthelp.sip"] + source_files,
                              include_dirs=include_dirs,
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args,
                              libraries=qt_libs,
                              library_dirs=library_dirs)


cmdclass = {
    "build_ext": build_pyqt_ext,
    "test": TestCommand,
}


def setup_package():
    setup(name=NAME,
          description=DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          packages=['pyqthelp'],
          url=URL,
          license=LICENSE,
          ext_modules=[pyqthelp_ext],
          cmdclass=cmdclass,
          install_requires=['numpy'],
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],)


if __name__ == '__main__':
    setup_package()
