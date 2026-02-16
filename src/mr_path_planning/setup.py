from setuptools import find_packages, setup
from glob import glob
import os

package_name = "mr_path_planning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config/rviz"), glob("config/rviz/*")),
        (os.path.join("share", package_name, "world"), glob("world/*.world")),
        (
            os.path.join("share", package_name, "world/include"),
            glob("world/include/*"),
        ),
        (
            os.path.join("share", package_name, "world/bitmaps"),
            glob("world/bitmaps/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Magraz",
    maintainer_email="agraz.manuel@outlook.com",
    description="TODO: Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "scan_max_to_inf = mr_path_planning.scan_max_to_inf:main",
        ],
    },
)
