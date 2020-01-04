from setuptools import setup, find_packages

requirements = [
    'toml', 'numpy'
]

setup(
    name='rcwa',
    version='0.0.1',
	packages=find_packages(),
    author='Gleb Siroki',
    author_email='g.shiroki@gmail.com',
    description='rcwa',
	install_requires = requirements,
	entry_points={
	    'console_scripts': [
		    'rcwa=rcwa.__main__:rcwa',
                    'tmm=rcwa.__main__:tmm'
		]
	},
)
