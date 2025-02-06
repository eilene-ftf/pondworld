from setuptools import setup

setup(
        name='pondworld',
        version='0.0.39',
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
                          'minigrid == 2.3.0', 
                          'tk',
                          'ipympl'
                          ],
        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
