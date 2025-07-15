"""Tests for the config module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from pyForwardFolding.config import (
    _build_components,
    _build_factors,
    _build_models,
    _load_config,
    analysis_from_config,
    models_from_config,
)


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_config_valid_file(self):
        """Test loading a valid YAML config file."""
        config_data = {
            "factors": [
                {
                    "name": "test_factor",
                    "type": "FluxNorm"
                }
            ],
            "components": [
                {
                    "name": "test_component",
                    "factors": ["test_factor"]
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "components": [
                        {
                            "name": "test_component",
                            "baseline_weight": "weight"
                        }
                    ]
                }
            ]
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = _load_config(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_config_nonexistent_file(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            _load_config("nonexistent_file.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                _load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestFactorBuilding:
    """Test factor building from configuration."""
    
    def test_build_factors_flux_norm(self):
        """Test building FluxNorm factors."""
        config = {
            "factors": [
                {
                    "name": "norm_factor",
                    "type": "FluxNorm"
                },
                {
                    "name": "norm_with_mapping",
                    "type": "FluxNorm",
                    "param_mapping": {
                        "flux_norm": "custom_norm"
                    }
                }
            ]
        }
        
        factors = _build_factors(config)
        
        assert len(factors) == 2
        assert "norm_factor" in factors
        assert "norm_with_mapping" in factors
        
        # Check factor properties
        norm_factor = factors["norm_factor"]
        assert norm_factor.name == "norm_factor"
        assert norm_factor.parameter_mapping["flux_norm"] == "flux_norm"
        
        norm_with_mapping = factors["norm_with_mapping"]
        assert norm_with_mapping.parameter_mapping["flux_norm"] == "custom_norm"
    
    def test_build_factors_powerlaw(self):
        """Test building PowerLawFlux factors."""
        config = {
            "factors": [
                {
                    "name": "powerlaw",
                    "type": "PowerLawFlux",
                    "pivot_energy": 1.0e5,
                    "baseline_norm": 1.0e-18,
                    "param_mapping": {
                        "flux_norm": "astro_norm",
                        "spectral_index": "astro_index"
                    }
                }
            ]
        }
        
        factors = _build_factors(config)
        
        assert len(factors) == 1
        powerlaw = factors["powerlaw"]
        assert powerlaw.name == "powerlaw"
        assert powerlaw.pivot_energy == 1.0e5
        assert powerlaw.baseline_norm == 1.0e-18
        assert powerlaw.parameter_mapping["flux_norm"] == "astro_norm"
        assert powerlaw.parameter_mapping["spectral_index"] == "astro_index"
    
    def test_build_factors_snowstorm_gauss(self):
        """Test building SnowstormGauss factors."""
        config = {
            "factors": [
                {
                    "name": "snowstorm",
                    "type": "SnowstormGauss",
                    "sys_gauss_width": 0.05,
                    "sys_sim_bounds": [0.9, 1.1],
                    "req_variable_name": "energy_scale",
                    "param_mapping": {
                        "scale": "escale"
                    }
                }
            ]
        }
        
        factors = _build_factors(config)
        
        snowstorm = factors["snowstorm"]
        assert snowstorm.name == "snowstorm"
        assert snowstorm.sys_gauss_width == 0.05
        assert snowstorm.sys_sim_bounds == [0.9, 1.1]
        assert "energy_scale" in snowstorm.required_variables
        assert snowstorm.parameter_mapping["scale"] == "escale"
    
    def test_build_factors_delta_gamma(self):
        """Test building DeltaGamma factors."""
        config = {
            "factors": [
                {
                    "name": "delta_gamma",
                    "type": "DeltaGamma",
                    "reference_energy": 2617.3148996675773
                }
            ]
        }
        
        factors = _build_factors(config)
        
        delta_gamma = factors["delta_gamma"]
        assert delta_gamma.name == "delta_gamma"
        assert delta_gamma.reference_energy == 2617.3148996675773
    
    def test_build_factors_unknown_type(self):
        """Test building factors with unknown type."""
        config = {
            "factors": [
                {
                    "name": "unknown_factor",
                    "type": "UnknownFactorType"
                }
            ]
        }
        
        with pytest.raises(KeyError):
            _build_factors(config)


class TestComponentBuilding:
    """Test component building from configuration."""
    
    def test_build_components_single_factor(self):
        """Test building components with single factor."""
        factors = {
            "norm_factor": type('MockFactor', (), {
                'name': 'norm_factor',
                'required_variables': [],
                'exposed_parameters': ['flux_norm'],
                'parameter_mapping': {'flux_norm': 'flux_norm'}
            })()
        }
        
        config = {
            "components": [
                {
                    "name": "test_component",
                    "factors": ["norm_factor"]
                }
            ]
        }
        
        components = _build_components(config, factors)
        
        assert len(components) == 1
        component = components["test_component"]
        assert component.name == "test_component"
        assert len(component.factors) == 1
        assert component.factors[0].name == "norm_factor"
    
    def test_build_components_multiple_factors(self):
        """Test building components with multiple factors."""
        factors = {
            "factor1": type('MockFactor1', (), {
                'name': 'factor1',
                'required_variables': ['var1'],
                'exposed_parameters': ['param1'],
                'parameter_mapping': {'param1': 'param1'}
            })(),
            "factor2": type('MockFactor2', (), {
                'name': 'factor2',
                'required_variables': ['var2'],
                'exposed_parameters': ['param2'],
                'parameter_mapping': {'param2': 'param2'}
            })()
        }
        
        config = {
            "components": [
                {
                    "name": "multi_factor_component",
                    "factors": ["factor1", "factor2"]
                }
            ]
        }
        
        components = _build_components(config, factors)
        
        component = components["multi_factor_component"]
        assert len(component.factors) == 2
        factor_names = [f.name for f in component.factors]
        assert "factor1" in factor_names
        assert "factor2" in factor_names
    
    def test_build_components_missing_factor(self):
        """Test building components with missing factor reference."""
        factors = {"existing_factor": type('MockFactor', (), {'name': 'existing_factor'})()}
        
        config = {
            "components": [
                {
                    "name": "test_component",
                    "factors": ["nonexistent_factor"]
                }
            ]
        }
        
        with pytest.raises(KeyError):
            _build_components(config, factors)


class TestModelBuilding:
    """Test model building from configuration."""
    
    def test_build_models_single_component(self):
        """Test building models with single component."""
        components = {
            "test_component": type('MockComponent', (), {
                'name': 'test_component',
                'required_variables': set(['var1']),
                'exposed_parameters': set(['param1']),
                'parameter_mapping': {'test_component': {'param1': 'param1'}}
            })()
        }
        
        config = {
            "models": [
                {
                    "name": "test_model",
                    "components": [
                        {
                            "name": "test_component",
                            "baseline_weight": "weight1"
                        }
                    ]
                }
            ]
        }
        
        models = _build_models(config, components)
        
        assert len(models) == 1
        model = models["test_model"]
        assert model.name == "test_model"
        assert len(model.components) == 1
        assert model.components[0].name == "test_component"
        assert model.baseline_weights == ["weight1"]
    
    def test_build_models_multiple_components(self):
        """Test building models with multiple components."""
        components = {
            "comp1": type('MockComponent1', (), {
                'name': 'comp1',
                'required_variables': set(['var1']),
                'exposed_parameters': set(['param1']),
                'parameter_mapping': {'comp1': {'param1': 'param1'}}
            })(),
            "comp2": type('MockComponent2', (), {
                'name': 'comp2',
                'required_variables': set(['var2']),
                'exposed_parameters': set(['param2']),
                'parameter_mapping': {'comp2': {'param2': 'param2'}}
            })()
        }
        
        config = {
            "models": [
                {
                    "name": "multi_component_model",
                    "components": [
                        {
                            "name": "comp1",
                            "baseline_weight": "weight1"
                        },
                        {
                            "name": "comp2",
                            "baseline_weight": "weight2"
                        }
                    ]
                }
            ]
        }
        
        models = _build_models(config, components)
        
        model = models["multi_component_model"]
        assert len(model.components) == 2
        assert model.baseline_weights == ["weight1", "weight2"]
        component_names = [c.name for c in model.components]
        assert "comp1" in component_names
        assert "comp2" in component_names


class TestIntegratedConfigFunctions:
    """Test the integrated config functions."""
    
    def test_models_from_config(self):
        """Test building models from complete config file."""
        config_data = {
            "factors": [
                {
                    "name": "powerlaw",
                    "type": "PowerLawFlux",
                    "pivot_energy": 1.0e5,
                    "baseline_norm": 1.0e-18,
                    "param_mapping": {
                        "flux_norm": "astro_norm",
                        "spectral_index": "astro_index"
                    }
                },
                {
                    "name": "atmo_norm",
                    "type": "FluxNorm",
                    "param_mapping": {
                        "flux_norm": "atmo_norm"
                    }
                }
            ],
            "components": [
                {
                    "name": "astro",
                    "factors": ["powerlaw"]
                },
                {
                    "name": "atmo",
                    "factors": ["atmo_norm"]
                }
            ],
            "models": [
                {
                    "name": "full_model",
                    "components": [
                        {
                            "name": "astro",
                            "baseline_weight": "astro_weight"
                        },
                        {
                            "name": "atmo",
                            "baseline_weight": "atmo_weight"
                        }
                    ]
                }
            ]
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            models = models_from_config(config_path)
            
            assert len(models) == 1
            model = models["full_model"]
            assert model.name == "full_model"
            assert len(model.components) == 2
            
            # Check that parameters are properly mapped
            expected_params = {"astro_norm", "astro_index", "atmo_norm"}
            assert model.exposed_parameters == expected_params
            
        finally:
            Path(config_path).unlink()
    
    def test_analysis_from_config_missing_expectations(self):
        """Test that analysis_from_config raises error for missing expectations."""
        config_data = {
            "factors": [
                {
                    "name": "norm_factor",
                    "type": "FluxNorm"
                }
            ],
            "components": [
                {
                    "name": "test_component",
                    "factors": ["norm_factor"]
                }
            ],
            "models": [
                {
                    "name": "test_model",
                    "components": [
                        {
                            "name": "test_component",
                            "baseline_weight": "weight"
                        }
                    ]
                }
            ]
            # Missing "expectations" section
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(KeyError):
                analysis_from_config(config_path)
        finally:
            Path(config_path).unlink()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_config_with_missing_factor_type(self):
        """Test config with missing factor type."""
        config = {
            "factors": [
                {
                    "name": "incomplete_factor"
                    # Missing "type" field
                }
            ]
        }
        
        with pytest.raises(KeyError):
            _build_factors(config)
    
    def test_config_with_missing_component_factors(self):
        """Test config with missing component factors."""
        config = {
            "components": [
                {
                    "name": "incomplete_component"
                    # Missing "factors" field
                }
            ]
        }
        
        with pytest.raises(KeyError):
            _build_components(config, {})
    
    def test_config_with_missing_model_components(self):
        """Test config with missing model components."""
        config = {
            "models": [
                {
                    "name": "incomplete_model"
                    # Missing "components" field
                }
            ]
        }
        
        with pytest.raises(KeyError):
            _build_models(config, {})
    
    def test_config_with_duplicate_names(self):
        """Test config with duplicate factor names."""
        config = {
            "factors": [
                {
                    "name": "duplicate_name",
                    "type": "FluxNorm"
                },
                {
                    "name": "duplicate_name",
                    "type": "PowerLawFlux",
                    "pivot_energy": 1e5,
                    "baseline_norm": 1e-18
                }
            ]
        }
        
        factors = _build_factors(config)
        # Second factor should overwrite the first due to dictionary behavior
        assert len(factors) == 1
        assert factors["duplicate_name"].__class__.__name__ == "PowerLawFlux"
