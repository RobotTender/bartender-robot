import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bartender_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fastcampus',
    maintainer_email='fastcampus@todo.todo',
    description='Custom Doosan robot control nodes',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gripper = bartender_test.gripper:main',
            'pour = bartender_test.pour:main',
            'pose = bartender_test.pose:main',
            'movej = bartender_test.movej:main',
            'movel = bartender_test.movel:main',
            'monitor = bartender_test.monitor:main',
            'snap = bartender_test.snap:main',
            'pick = bartender_test.pick:main',
        ],
    },
)
