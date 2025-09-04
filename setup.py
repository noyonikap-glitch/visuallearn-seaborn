from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
def get_version():
    with open(os.path.join("visuallearn", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.2.0"

setup(
    name='visuallearn',
    version=get_version(),
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'matplotlib>=3.3.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.23.0',
        'imageio>=2.8.0',
        'Pillow>=7.0.0',
    ],
    extras_require={
        'pytorch': ['torch>=1.6.0'],
        'video': ['imageio[ffmpeg]'],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
        ],
        'all': ['torch>=1.6.0', 'imageio[ffmpeg]'],
    },
    author='Noyonika Puram',
    author_email='noyonikapuram@gmail.com',
    description='A Python library for visualizing ML model learning patterns and decision boundaries in real-time',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/noyonikap-glitch/visuallearn',
    project_urls={
        'Bug Reports': 'https://github.com/noyonikap-glitch/visuallearn/issues',
        'Source': 'https://github.com/noyonikap-glitch/visuallearn',
        'Documentation': 'https://github.com/noyonikap-glitch/visuallearn#readme',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning, visualization, decision boundaries, neural networks, deep learning, education',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
)
