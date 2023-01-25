from setuptools import setup

setup(
        name='pondworld',
        version='0.0.6',
        description='A little minigrid world featuring a frog that eats flies',
        author='Eilene Tomkins-Flanagan',
        author_email='eilenetomkinsflanaga@cmail.carleton.ca',
        packages=['pondworld', 'pondworld.envs'],
        install_requires=['gymnasium', 'numpy', 'cairosvg', 'opencv-python', 'minigrid', 'tk'],

        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
