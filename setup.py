from setuptools import setup

setup(
        name='pondworld',
        version='0.0.13',
        description='A little minigrid world featuring a frog that eats flies',
        author='Eilene Tomkins-Flanagan',
        author_email='eilenetomkinsflanaga@cmail.carleton.ca',
        include_package_data=True,
        packages=['pondworld', 'pondworld.envs', 'assets'],
        install_requires=['gymnasium', 
                          'numpy', 
                          'cairosvg', 
                          'opencv-python', 
                          'minigrid', 
                          'tk',
                          'ipympl'],

        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
