"""Tests for version and package integrity."""

import pytest

import pyForwardFolding


class TestVersion:
    """Test package version information."""
    
    def test_version_exists(self):
        """Test that version attribute exists."""
        assert hasattr(pyForwardFolding, '__version__')
    
    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        version = pyForwardFolding.__version__
        assert isinstance(version, str)
        
        # Basic semantic version check (major.minor.patch)
        parts = version.split('.')
        assert len(parts) >= 2  # At least major.minor
        
        # First two parts should be integers
        assert parts[0].isdigit()
        assert parts[1].isdigit()
    
    def test_version_not_empty(self):
        """Test that version is not empty."""
        version = pyForwardFolding.__version__
        assert version.strip() != ""


class TestPackageStructure:
    """Test package structure and imports."""
    
    def test_main_modules_importable(self):
        """Test that main modules can be imported."""
        # Test importing main modules
        from pyForwardFolding import analysis
        from pyForwardFolding import backend
        from pyForwardFolding import binned_expectation
        from pyForwardFolding import binning
        from pyForwardFolding import buffers
        from pyForwardFolding import clustering
        from pyForwardFolding import config
        from pyForwardFolding import factor
        from pyForwardFolding import likelihood
        from pyForwardFolding import minimizer
        from pyForwardFolding import model
        from pyForwardFolding import model_component
        
        # Basic smoke test - check that modules have expected attributes
        assert hasattr(analysis, 'Analysis')
        assert hasattr(backend, 'Backend')
        assert hasattr(backend, 'JAXBackend')
        assert hasattr(binning, 'AbstractBinning')
        assert hasattr(buffers, 'BufferManager')
        assert hasattr(factor, 'AbstractFactor')
        assert hasattr(likelihood, 'AbstractLikelihood')
        assert hasattr(model, 'Model')
        assert hasattr(model_component, 'ModelComponent')
    
    def test_key_classes_accessible(self):
        """Test that key classes are accessible from main package."""
        # These are the most commonly used classes
        from pyForwardFolding.analysis import Analysis
        from pyForwardFolding.backend import JAXBackend
        from pyForwardFolding.binning import RectangularBinning
        from pyForwardFolding.buffers import BufferManager
        from pyForwardFolding.factor import PowerLawFlux, FluxNorm
        from pyForwardFolding.likelihood import PoissonLikelihood
        from pyForwardFolding.model import Model
        from pyForwardFolding.model_component import ModelComponent
        
        # Basic instantiation test (should not raise errors)
        backend = JAXBackend()
        assert backend is not None
        
        # Test that classes exist and are callable
        assert callable(Analysis)
        assert callable(RectangularBinning)
        assert callable(BufferManager)
        assert callable(PowerLawFlux)
        assert callable(FluxNorm)
        assert callable(PoissonLikelihood)
        assert callable(Model)
        assert callable(ModelComponent)
    
    def test_config_functions_accessible(self):
        """Test that config functions are accessible."""
        from pyForwardFolding.config import (
            analysis_from_config,
            models_from_config,
        )
        
        # Functions should be callable
        assert callable(analysis_from_config)
        assert callable(models_from_config)
    
    def test_minimizer_functions_accessible(self):
        """Test that minimizer functions are accessible."""
        from pyForwardFolding.minimizer import (
            destructure_args,
            flat_index_dict_mapping,
            restructure_args,
            WrappedLLH,
        )
        
        # Functions should be callable
        assert callable(destructure_args)
        assert callable(flat_index_dict_mapping)
        assert callable(restructure_args)
        assert callable(WrappedLLH)


class TestPackageConstants:
    """Test package constants and default values."""
    
    def test_backend_default_instance(self):
        """Test that default backend instance exists."""
        from pyForwardFolding.backend import backend
        
        # Should be a JAXBackend instance
        assert backend is not None
        assert hasattr(backend, 'array')
        assert hasattr(backend, 'zeros')
        assert callable(backend.array)
    
    def test_factor_registry(self):
        """Test that factor class registry exists."""
        from pyForwardFolding.factor import FACTORSTR_CLASS_MAPPING
        
        # Should contain key factor types
        assert isinstance(FACTORSTR_CLASS_MAPPING, dict)
        assert "PowerLawFlux" in FACTORSTR_CLASS_MAPPING
        assert "FluxNorm" in FACTORSTR_CLASS_MAPPING
        assert "SnowstormGauss" in FACTORSTR_CLASS_MAPPING
        assert "DeltaGamma" in FACTORSTR_CLASS_MAPPING
        
        # All values should be classes
        for factor_class in FACTORSTR_CLASS_MAPPING.values():
            assert isinstance(factor_class, type)


class TestPackageIntegrity:
    """Test overall package integrity."""
    
    def test_no_import_errors(self):
        """Test that importing the package doesn't raise errors."""
        # This test is implicitly run by importing pyForwardFolding at the top
        # But let's be explicit about it
        try:
            import pyForwardFolding
            # Try importing all submodules
            import pyForwardFolding.analysis
            import pyForwardFolding.backend
            import pyForwardFolding.binned_expectation
            import pyForwardFolding.binning
            import pyForwardFolding.buffers
            import pyForwardFolding.clustering
            import pyForwardFolding.config
            import pyForwardFolding.factor
            import pyForwardFolding.likelihood
            import pyForwardFolding.minimizer
            import pyForwardFolding.model
            import pyForwardFolding.model_component
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
    
    def test_basic_workflow_imports(self):
        """Test that a basic workflow can import all necessary components."""
        import importlib.util
        
        # Check that workflow modules are available
        workflow_modules = [
            'pyForwardFolding.config',
            'pyForwardFolding.likelihood', 
            'pyForwardFolding.minimizer',
            'pyForwardFolding.factor',
            'pyForwardFolding.model',
            'pyForwardFolding.model_component',
            'pyForwardFolding.binning',
            'pyForwardFolding.analysis'
        ]
        
        for module_name in workflow_modules:
            spec = importlib.util.find_spec(module_name)
            assert spec is not None, f"Workflow module '{module_name}' not available"
    
    def test_dependencies_available(self):
        """Test that required dependencies are available."""
        import importlib.util
        
        required_deps = ['jax', 'numpy', 'pandas', 'yaml', 'iminuit']
        
        for dep in required_deps:
            spec = importlib.util.find_spec(dep)
            assert spec is not None, f"Required dependency '{dep}' not found"


class TestPackageMetadata:
    """Test package metadata."""
    
    def test_package_has_docstring(self):
        """Test that main modules have docstrings."""
        import pyForwardFolding.analysis
        import pyForwardFolding.backend
        import pyForwardFolding.factor
        import pyForwardFolding.model
        
        # Not all modules may have module-level docstrings, 
        # but key classes should have docstrings
        assert pyForwardFolding.analysis.Analysis.__doc__ is not None
        assert pyForwardFolding.backend.Backend.__doc__ is not None
        assert pyForwardFolding.factor.AbstractFactor.__doc__ is not None
        assert pyForwardFolding.model.Model.__doc__ is not None
    
    def test_typed_package(self):
        """Test that package declares itself as typed."""
        # Package should have a py.typed file
        import pyForwardFolding
        package_dir = pyForwardFolding.__path__[0]
        
        from pathlib import Path
        py_typed_file = Path(package_dir) / "py.typed"
        
        # py.typed file should exist (indicates type information is available)
        assert py_typed_file.exists(), "Package should have py.typed file for type hints"
