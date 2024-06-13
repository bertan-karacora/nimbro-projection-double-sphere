import setuptools

NAME_PACKAGE = "nimbro_projection_double_sphere"

setuptools.setup(
    name=NAME_PACKAGE,
    version="0.0.1",
    packages=setuptools.find_namespace_packages(exclude=["resource"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{NAME_PACKAGE}"]),
        (f"share/{NAME_PACKAGE}", ["package.xml"]),
    ],
    package_data={NAME_PACKAGE: ["configs/*.cset"]},
    include_package_data=True,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bertan Karacora",
    maintainer_email="bertan.karacora@gmail.com",
    description="ROS2 package for image-to-point-cloud and point-cloud-to-image projection",
    license="MIT",
    entry_points={
        "console_scripts": [
            f"spin = {NAME_PACKAGE}.scripts.spin_projection:main",
        ],
    },
)
