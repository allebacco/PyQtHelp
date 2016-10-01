import os
import subprocess
import sys

from setuptools import Command


class TestCommand(Command):

    user_options = []

    def initialize_options(self):
        self.test_build_dir = 'build_for_tests'

    def finalize_options(self):
        pass

    def _build_extension(self):
        build_command = [
            sys.executable, 'setup.py', 'build',
            '--build-base', self.test_build_dir,
            '--build-lib', self.test_build_dir,
        ]
        ok = subprocess.call(build_command)
        if ok != 0:
            raise RuntimeError('Unable to build the library for testing')

    def run(self):
        import pytest

        self._build_extension()

        # Add build directory in the first position of PYTHON_PATH
        current_dir = os.path.abspath(os.getcwd())
        new_path_dir = os.path.join(current_dir, self.test_build_dir)
        sys.path.insert(0, new_path_dir)

        ret = pytest.main([])

        sys.exit(ret)

