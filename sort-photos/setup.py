import os


from setuptools import setup, find_packages

setup(
    name="clip-photo-sorter",
    version="1.0",
    description="Sort café photos into Food & Drinks, People and Interior albums with a frozen CLIP encoder",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        if line.strip() and not line.startswith("#")
    ],
    entry_points={"console_scripts": ["photo-sorter=clip.cli:main"]},
    extras_require={'dev': ['pytest==9.2.0']},
)
