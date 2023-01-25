from setuptools import setup

setup(
        name='pondworld',
        version='0.0.3',
        description='A little minigrid world featuring a frog that eats flies',
        author='Eilene Tomkins-Flanagan',
        author_email='eilenetomkinsflanaga@cmail.carleton.ca',
        packages=['gym', 'numpy', 'cairosvg', 'opencv-python', 'minigrid'],

        classifiers=[
            'Programming Language :: Python :: 3'
            ]
        )
