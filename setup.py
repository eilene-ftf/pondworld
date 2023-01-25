from setuptools import setup

setup(
        name='pondworld',
        version='0.0.25',
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
                          'gymnasium', 
                          'numpy',
                          'cairosvg', 
                          'opencv-python', 
                          'minigrid', 
                          'tk',
                          'ipympl'
                          ],
        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
