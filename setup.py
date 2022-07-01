from setuptools import find_packages, setup

setup(name='braid_mrf',
      version='1.0.5',
      description='Predicting protein complexes',
      author='Wasinee Rungsarityotin',
      author_email='wasinees@gmail.com',
      url='https://github.com/wasineer-dev/braid.git',
      packages=['braidmrf', 'braidmrf.meanfield', 'braidmrf.inputFile', 'braidmrf.observation'],
     )