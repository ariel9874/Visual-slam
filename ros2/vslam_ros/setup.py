from setuptools import setup

package_name = "vslam_ros"

setup(
    name=package_name,
    version="0.8.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/tum_demo.launch.py",
                                               "launch/euroc_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ariel Vazquez",
    maintainer_email="ariel98745@gmail.com",
    description="Cascara ROS 2 del Visual-SLAM educativo (nucleo en vslam/).",
    license="TBD",
    entry_points={
        "console_scripts": [
            "dataset_node = vslam_ros.dataset_node:main",
            "frontend_node = vslam_ros.frontend_node:main",
            "backend_node = vslam_ros.backend_node:main",
            "mapper_node = vslam_ros.mapper_node:main",
        ],
    },
)
