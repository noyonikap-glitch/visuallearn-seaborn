# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-04

### Added
- Complete package restructuring for PyPI readiness
- Proper `__init__.py` with all exports and version information
- Comprehensive utility functions in `utils.py`
- MIT License file
- MANIFEST.in for proper package inclusion
- pyproject.toml for modern Python packaging
- Enhanced setup.py with proper metadata and dependencies
- Quick start example demonstrating library usage
- Type hints and improved error handling
- Support for optional dependencies (PyTorch, video export)

### Changed
- Updated package version from 0.1 to 0.2.0
- Improved README with better installation instructions
- Enhanced error messages and validation
- Better matplotlib backend handling

### Fixed
- Missing imports in package initialization
- Package structure for proper PyPI distribution
- Dependencies management with extras_require

## [0.1.0] - 2024-05-12

### Added
- Initial release with basic functionality
- Decision boundary visualization
- Activation and gradient tracking for PyTorch models
- Real-time loss plotting
- GIF and MP4 export capabilities
- Support for scikit-learn and PyTorch models
- Combined plotting coordinator
- CNN feature map visualization (experimental)