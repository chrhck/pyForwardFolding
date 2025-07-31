from typing import Dict, Set, Tuple, Union

from .backend import Array
from .binned_expectation import BinnedExpectation


class Analysis:
    """
    Represents a complete analysis consisting of multiple binned expectations.

    Args:
        expectations (Dict[str, BinnedExpectation]): Dictionary mapping expectation names to their objects.
    """

    def __init__(self, expectations: Dict[str, BinnedExpectation]):
        self.expectations = expectations

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the analysis.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return set.union(
            *(comp.required_variables for comp in self.expectations.values())
        )

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get all parameters exposed by the analysis expectations.

        Returns:
            Set[str]: A set of all exposed parameters from all expectations.
        """
        exposed = set()
        for comp in self.expectations.values():
            exposed |= comp.exposed_parameters
        return exposed

    def evaluate(
        self,
        datasets: Dict[str, Dict[str, Union[Array, float]]],
        parameter_values: Dict[str, float],
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        """
        Evaluate all expectations in the analysis.

        Args:
            datasets (Dict[str, Dict[str, Union[Array, float]]]): A dictionary mapping component names to their input variables.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated expectations.

        Returns:
            Tuple[Dict[str, Array], Dict[str, Array]]: A tuple containing:
                - A dictionary mapping component names to their evaluation results (histograms).
                - A dictionary mapping component names to their squared evaluation results (histograms).
        """
        output_dict = {}
        output_ssq_dict = {}

        for comp_name, comp in self.expectations.items():
            # Evaluate the component
            hist, hist_ssq = comp.evaluate(
                datasets,
                parameter_values,
            )

            # Store results
            output_dict[comp_name] = hist
            output_ssq_dict[comp_name] = hist_ssq

        return output_dict, output_ssq_dict

    def __repr__(self):
        """
        String representation of the Analysis object.

        Returns:
            str: A comprehensive string representation of the analysis.
        """
        lines = []
        lines.append(f"Analysis with {len(self.expectations)} expectations")
        lines.append("=" * 50)
        
        # Overall summary
        lines.append(f"Required variables: {sorted(self.required_variables)}")
        lines.append(f"Exposed parameters: {sorted(self.exposed_parameters)}")
        lines.append("")
        
        # Detailed expectation information
        for exp_name, expectation in self.expectations.items():
            lines.append(f"Expectation: {exp_name}")
            lines.append("-" * 30)
            
            # Dataset keys and models
            dataset_keys = [dskey for dskey, _ in expectation.dskey_model_pairs]
            lines.append(f"  Dataset keys: {dataset_keys}")
            
            # Models and components with detailed information
            lines.append(f"  Models ({len(expectation.models)}):")
            for model in expectation.models:
                # Use the model's __repr__ method and indent it
                model_repr = str(model).replace('\n', '\n    ')
                lines.append(f"    {model_repr}")
            
            # Binning information using its __repr__ method
            lines.append("  Binning:")
            binning_repr = str(expectation.binning).replace('\n', '\n    ')
            lines.append(f"    {binning_repr}")
            
            # Binned factors with detailed information
            if expectation.binned_factors:
                lines.append(f"  Binned factors ({len(expectation.binned_factors)}):")
                for factor in expectation.binned_factors:
                    factor_repr = str(factor).replace('\n', '\n    ')
                    lines.append(f"    {factor_repr}")
            
            # Variables and parameters for this expectation
            lines.append(f"  Required variables: {sorted(expectation.required_variables)}")
            lines.append(f"  Exposed parameters: {sorted(expectation.exposed_parameters)}")
            lines.append(f"  Lifetime: {expectation.lifetime}")
            lines.append("")
        
        return "\n".join(lines)

    def _repr_markdown_(self):
        """
        Markdown representation for Jupyter notebooks.

        Returns:
            str: A markdown-formatted string representation of the analysis.
        """
        return self.repr_markdown()

    def repr_markdown(
        self,
        indent_level: int = 0,
        bullet_style: str = "-",
        include_summary: bool = True,
        show_expectation_details: bool = True,
        show_model_details: bool = True,
        show_binning_details: bool = True,
    ) -> str:
        """
        Configurable markdown representation of the Analysis object.

        Args:
            indent_level (int): The level of indentation (each level adds 2 spaces). Default is 0.
            bullet_style (str): Style for bullet points ("-", "*", "+"). Default is "-".
            include_summary (bool): Whether to include overall summary information. Default is True.
            show_expectation_details (bool): Whether to show detailed expectation information. Default is True.
            show_model_details (bool): Whether to show detailed model information. Default is True.
            show_binning_details (bool): Whether to show detailed binning information. Default is True.

        Returns:
            str: A configurable markdown representation of the analysis.
        """
        indent = "  " * indent_level
        sub_indent = "  " * (indent_level + 1)
        
        lines = []
        
        # Header with appropriate level based on indent
        if indent_level == 0:
            lines.append("# Analysis Configuration")
            lines.append(f"**{len(self.expectations)} expectations configured**")
        else:
            lines.append(f"{indent}{bullet_style} **Analysis Configuration**")
            lines.append(f"{sub_indent}{bullet_style} {len(self.expectations)} expectations configured")
        
        lines.append("")

        # Overall summary
        if include_summary:
            if indent_level == 0:
                lines.append("## Summary")
                lines.append("| Aspect | Details |")
                lines.append("|--------|---------|")
                lines.append(f"| Required variables | `{', '.join(sorted(self.required_variables))}` |")
                lines.append(f"| Exposed parameters | `{', '.join(sorted(self.exposed_parameters))}` |")
            else:
                lines.append(f"{sub_indent}{bullet_style} Summary:")
                summary_indent = "  " * (indent_level + 2)
                lines.append(f"{summary_indent}{bullet_style} Required variables: `{', '.join(sorted(self.required_variables))}`")
                lines.append(f"{summary_indent}{bullet_style} Exposed parameters: `{', '.join(sorted(self.exposed_parameters))}`")
            lines.append("")

        # Detailed expectation information
        if show_expectation_details:
            for exp_name, expectation in self.expectations.items():
                exp_md = expectation.repr_markdown(
                    indent_level=indent_level + 1 if indent_level == 0 else indent_level + 2,
                    bullet_style=bullet_style,
                    include_summary=include_summary,
                    show_model_details=show_model_details,
                    show_binning_details=show_binning_details,
                )
                lines.append(exp_md)
                lines.append("")
        else:
            # Just list expectation names
            if indent_level == 0:
                lines.append("## Expectations:")
            else:
                lines.append(f"{sub_indent}{bullet_style} Expectations:")
            
            for exp_name in self.expectations.keys():
                exp_indent = sub_indent if indent_level == 0 else "  " * (indent_level + 2)
                lines.append(f"{exp_indent}{bullet_style} `{exp_name}`")
            lines.append("")

        return "\n".join(lines)

    def __getitem__(self, item: str) -> BinnedExpectation:
        """
        Get a specific expectation by name.

        Args:
            item (str): The name of the expectation.

        Returns:
            BinnedExpectation: The corresponding expectation object.
        """
        return self.expectations[item]
