from setuptools import find_packages, setup

package_name = 'dsr_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gossi',
    maintainer_email='mincheol710313@gmail.com',
    description='TODO: Package description',
    license='BSD',
    entry_points={
        'console_scripts': [
                'gripper_control =        dsr_example.simple.gripper_control:main',
                'gripper_drl_controller = dsr_example.simple.gripper_drl_controller:main',
                'new_gripper_drl_controller = dsr_example.simple.new_gripper_drl_controller:main',
                'example_gripper = dsr_example.simple.example_gripper:main',

                'dance =               dsr_example.demo.dance_m1013:main',
                'slope_demo =          dsr_example.demo.slope_demo:main',
                
                'single_robot_simple = dsr_example.simple.single_robot_simple:main',
                'movel_test =          dsr_example.simple.movel_test:main',

                'gripper_test =     dsr_example.isaacsim.gripper_test:main',
                'isaac_e0509_gripper =           dsr_example.isaacsim.isaac_e0509_gripper:main',
                'pick_and_place =      dsr_example.isaacsim.pick_and_place:main',                 
                'move_gripper =     dsr_example.isaacsim.move_gripper:main',
        ],
    },
)
