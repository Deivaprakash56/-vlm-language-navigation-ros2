from glob import glob
from setuptools import setup

package_name = 'lang_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deiva',
    description='Language-guided navigation with a VLM and Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lang_nav_node = lang_nav.lang_nav_node:main',
            'lang_nav_cli = lang_nav.cli:main',
            'set_initial_pose = lang_nav.set_initial_pose:main',
            'demo_eval = lang_nav.demo_eval:main',
            'plot_demo = lang_nav.plot_demo:main',
        ],
    },
)
