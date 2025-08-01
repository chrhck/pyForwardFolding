from typing import Dict, List, Optional, Set, Tuple, Union

from .backend import Array, backend
from .binning import AbstractBinning
from .factor import AbstractBinnedFactor
from .model import Model


class BinnedExpectation:
    """
    Represents an expectation calculated over binned data.

    Args:
        name (str): Name of the BinnedExpectation.
        dskey_model_pairs (List[Tuple[str, Model]]): List of tuples where each tuple contains a dataset key and a model.
        binning (AbstractBinning): The binning used to create the histogram.
        binned_factors (Optional[List[AbstractBinnedFactor]]): Factors to be added to the histogram.
        lifetime (float): Lifetime of the binned expectation.
    """

    def __init__(
        self,
        name: str,
        dskey_model_pairs: List[Tuple[str, Model]],
        binning: AbstractBinning,
        binned_factors: Optional[List[AbstractBinnedFactor]] = None,
        lifetime: float = 1.0,
    ):
        self.name = name
        self.dskey_model_pairs = dskey_model_pairs
        self.models = [model for _, model in dskey_model_pairs]
        self.binning = binning
        self.binned_factors = binned_factors if binned_factors is not None else []
        self.lifetime = lifetime

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by the BinnedExpectation.

        Returns:
        """

        model_required_vars = set()
        for model in self.models:
            model_required_vars.update(model.required_variables)

        return set(self.binning.required_variables).union(model_required_vars)

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get parameters exposed by the BinnedExpectation.

        Returns:
            Set[str]: A set of all exposed parameters from models and binned factors.
        """

        model_exposed = set()

        for model in self.models:
            model_exposed.update(model.exposed_parameters)

        bf_exposed = {
            par for factor in self.binned_factors for par in factor.exposed_parameters
        }

        return model_exposed | bf_exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
    ) -> Tuple[Array, Array]:
        """
        Evaluate a binned expectation by creating a weighted histogram.

        Args:
            datasets (Dict[str, Dict[str, Union[Array, float]]]): A dictionary where keys are dataset names and values are dictionaries of input variables.
            parameter_values (Dict[str, float]): A dictionary of parameter values, where keys are parameter names and values are floats.

        Returns:
            Tuple[Array, Array]: A tuple containing:
                - The histogram weights (Array).
                - The squared weights (Array) representing the binned expectation.
        """

        hist = backend.zeros(self.binning.hist_dims)
        hist_ssq = backend.zeros(self.binning.hist_dims)

        for model_dskey, model in self.dskey_model_pairs:
            if model_dskey not in datasets:
                raise ValueError(
                    f"Dataset '{model_dskey}' not found in provided datasets."
                )
            input_variables = datasets[model_dskey]

            # Evaluate the model to get weights
            weights = model.evaluate(
                input_variables,
                parameter_values,
            )
            weight_sq = backend.power(weights, 2)
            # Extract binning variables
            binning_variables = tuple(
                backend.asarray(input_variables[var])
                for var in self.binning.required_variables
            )

            # Build histograms
            hist += (
                self.binning.build_histogram(model_dskey, weights, binning_variables)
                * self.lifetime
            )
            hist_ssq += (
                self.binning.build_histogram(model_dskey, weight_sq, binning_variables)
                * self.lifetime**2
            )

            # Add contributions from binned factors
            for factor in self.binned_factors:
                # TODO: Update logic. Do we want to store the bin edges in the factor?

                # if isinstance(self.binning, RectangularBinning):
                #     if factor.bin_edges is None:
                #         raise ValueError("Binned factors must have bin_edges defined.")
                #     else:
                #         for j, bin_edge in enumerate(self.binning.bin_edges):
                #             print("Set binning", bin_edge)
                #             print("Loaded Binning", factor.bin_edges[self.det_config][j])
                #             np.testing.assert_array_almost_equal(
                #                 bin_edge,
                #                 factor.bin_edges[self.det_config][j],
                #                 err_msg=f"Binned factor {factor.name} has different bin edges than the binning."
                #             )
                hist_add, hist_ssq_add = factor.evaluate(
                    input_variables,
                    parameter_values,
                )
                hist += hist_add * self.lifetime
                if hist_ssq_add is not None:
                    hist_ssq += hist_ssq_add * self.lifetime**2

        # Clamp values to avoid negative or infinite values
        hist = backend.clip(hist, 0, float("inf"))
        hist_ssq = backend.clip(hist_ssq, 0, float("inf"))

        return hist, hist_ssq

    def _repr_markdown_(self) -> str:
        """
        Markdown representation of the BinnedExpectation object.

        Returns:
            str: A markdown representation of the binned expectation.
        """
        return self.repr_markdown()

    def repr_markdown(
        self,
        indent_level: int = 0,
        bullet_style: str = "-",
        include_summary: bool = True,
        show_model_details: bool = True,
        show_binning_details: bool = True,
    ) -> str:
        """
        Configurable markdown representation of the BinnedExpectation object.

        Args:
            indent_level (int): The level of indentation (each level adds 2 spaces). Default is 0.
            bullet_style (str): Style for bullet points ("-", "*", "+"). Default is "-".
            include_summary (bool): Whether to include summary information. Default is True.
            show_model_details (bool): Whether to show detailed model information. Default is True.
            show_binning_details (bool): Whether to show detailed binning information. Default is True.

        Returns:
            str: A configurable markdown representation of the binned expectation.
        """
        indent = "  " * indent_level
        sub_indent = "  " * (indent_level + 1)
        
        lines = []
        
        # Header with appropriate level based on indent
        if indent_level == 0:
            lines.append(f"## Expectation: `{self.name}`")
        else:
            lines.append(f"{indent}{bullet_style} **Expectation:** `{self.name}`")
        
        # Dataset keys
        dataset_keys = [dskey for dskey, _ in self.dskey_model_pairs]
        if indent_level == 0:
            lines.append(f"**Datasets:** `{dataset_keys}`")
        else:
            lines.append(f"{sub_indent}{bullet_style} Datasets: `{dataset_keys}`")
        
        lines.append("")
        
        # Models section
        if indent_level == 0:
            lines.append(f"**Models ({len(self.models)}):**")
        else:
            lines.append(f"{sub_indent}{bullet_style} Models ({len(self.models)}):")
        
        if show_model_details:
            # Create a mapping from model to datasets
            model_to_datasets = {}
            for dskey, model in self.dskey_model_pairs:
                if model.name not in model_to_datasets:
                    model_to_datasets[model.name] = []
                model_to_datasets[model.name].append(dskey)
            
            for i, model in enumerate(self.models, 1):
                datasets_used = model_to_datasets.get(model.name, [])
                datasets_str = ', '.join(f"`{ds}`" for ds in datasets_used)
                
                model_indent = "" if indent_level == 0 else sub_indent
                lines.append(f"{model_indent}{i}. **{model.name}** ({len(model.components)} components) - Uses datasets: {datasets_str}")
                
                
                # Use compact model representation
                model_md = model.repr_markdown(
                    indent_level=indent_level + 1 if indent_level == 0 else indent_level + 2,
                    bullet_style=bullet_style,
                    include_summary=False,  # Skip model summary to avoid redundancy
                    show_component_details=False,  # Keep it compact
                )
                lines.append(model_md)           
        else:
            # Just list model names
            for model in self.models:
                model_indent = sub_indent if indent_level == 0 else "  " * (indent_level + 2)
                lines.append(f"{model_indent}{bullet_style} {model.name}")
        
        lines.append("")
        
        # Binning information
        if show_binning_details:
            if indent_level == 0:
                lines.append("**Binning:**")
            else:
                lines.append(f"{sub_indent}{bullet_style} Binning:")
            
            binning_indent = sub_indent if indent_level == 0 else "  " * (indent_level + 2)
            lines.append(f"{binning_indent}{bullet_style} Type: `{type(self.binning).__name__}`")
            lines.append(f"{binning_indent}{bullet_style} Variables: `{self.binning.required_variables}`")
            lines.append(f"{binning_indent}{bullet_style} Dimensions: `{self.binning.hist_dims}` ({self.binning.nbins} bins)")
            lines.append("")
        
        # Binned factors if any
        if self.binned_factors:
            factor_names = [f.name for f in self.binned_factors]
            if indent_level == 0:
                lines.append(f"**Binned Factors:** {', '.join(factor_names)}")
            else:
                lines.append(f"{sub_indent}{bullet_style} Binned Factors: {', '.join(factor_names)}")
            lines.append("")
        
        # Summary information
        if include_summary:
            if indent_level == 0:
                lines.append("**Configuration Summary:**")
                lines.append("| Aspect | Count/Details |")
                lines.append("|--------|---------------|")
                lines.append(f"| Required variables | {len(self.required_variables)} |")
                lines.append(f"| Exposed parameters | {len(self.exposed_parameters)} |")
                lines.append(f"| Lifetime | `{self.lifetime}` |")
            else:
                lines.append(f"{sub_indent}{bullet_style} Required variables: {len(self.required_variables)}")
                lines.append(f"{sub_indent}{bullet_style} Exposed parameters: {len(self.exposed_parameters)}")
                lines.append(f"{sub_indent}{bullet_style} Lifetime: `{self.lifetime}`")

        return "\n".join(lines)
