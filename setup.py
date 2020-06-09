from setuptools import setup, find_packages
setup(
    name="gigaanalysis",
    version="0.01",
    packages=find_packages(),
    # scripts=["data.py", "qo.py"],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["numpy", "scipy", "pandas", "matplotlib"],

    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     "": ["*.txt", "*.rst"],
    #     # And include any *.msg files found in the "hello" package, too:
    #     "hello": ["*.msg"],
    # },

    # metadata to display on PyPI
    author="Alexander J Hickey",
    author_email="alexander.john.hickey@gmail.com",
    description="A toolbox for manipulating high field experimental data.",
    keywords="magnets SdH dHvA magnetic PPMS MPMS quantum" \
        "oscillations superconductor Tesla",
    # url="",   # project home page, if any
    # project_urls={
    #     "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #     "Documentation": "https://docs.example.com/HelloWorld/",
    #     "Source Code": "https://code.example.com/HelloWorld/",
    # },
    # classifiers=[
    #     "License :: OSI Approved :: Python Software Foundation License"
    #     ]

    # could also include long_description, download_url, etc.
)
