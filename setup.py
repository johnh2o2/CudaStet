import  os
from os.path import join as pjoin
import  subprocess
from setuptools import setup
from distutils.extension import Extension

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

py_dir = 'cudastet'
wrap_file = os.path.join(py_dir, 'cudastet_wrap.cpp')

# check for swig
if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -python -c++ -o {wfile} -I./inc\
 {pdir}/cudastet.i'.format(wfile=wrap_file, pdir=py_dir), shell=True)
else:
    raise EnvironmentError('the swig executable was not found in your PATH')

VERSION=open('VERSION.txt', 'r').read().strip('\n')
macros = dict(VERSION=VERSION)
sources = [wrap_file]

ext = Extension('_cudastet',
		define_macros=[ ( key, value ) for key, value in macros.iteritems() ],
                sources=sources,
                libraries=['m', 'custet'],
                language='c++',
                extra_compile_args= [ '-fPIC', '-O3' ] ,
                include_dirs = [ '/usr/local/include/custet' ])


setup(name='cudastet',

      # random metadata. there's more you can supploy
      author='John Hoffman',
      author_email='jah5@princeton.edu',
      description='Fast CUDA Stetson variability index module',
      version=VERSION,
      license='COPYING.txt',

      py_modules=['cudastet'],
      package_dir={'' : 'cudastet'},
      ext_modules = [ext],

      # since the package has c code, the egg cannot be zipped
      zip_safe=False)
