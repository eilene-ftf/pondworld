from setuptools import setup

setup(
        name='pondworld',
        version='0.0.38',
        description='A little minigrid world featuring a frog that eats flies',
        author='Eilene Tomkins-Flanagan',
        author_email='eilenetomkinsflanaga@cmail.carleton.ca',
        include_package_data=True,
        packages=[
            'pondworld', 
            'pondworld.envs', 
            'pondworld.frog_control', 
            'pond_assets'
            ],
        install_requires=[
                          'gymnasium == 0.29.0', 
                          'numpy',
                          'cairosvg', 
                          'opencv-python', 
                          'minigrid == 2.1.1', 
                          'tk',
                          'ipympl'
                          ],
        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
