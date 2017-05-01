from setuptools import setup, find_packages

setup(name='core_overshoot_clusters',
      version='0.1',
      description='Repo for reproducing Rosenfield et. al ApJ 2017, xxx, xxx',
      url='http://github.com/philrosenfield/core_overshoot_clusters',
      author='Philip Rosenfield',
      author_email='philip.rosenfield@cfa.harvard.edu',
      license='MIT',
      packages=['core_overshoot_clusters'],
      zip_safe=False,
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4'],
      install_requires=['matplotlib', 'numpy', 'astropy', 'scipy', 'pandas'],
      include_package_data=True)
