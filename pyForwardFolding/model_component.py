from typing import Dict, List, Set, Union

from .backend import Array, backend
from .factor import AbstractUnbinnedFactor


class ModelComponent:
    """
    A component of a model that contains multiple factors.

    Args:
        name (str): The name of the component.
        factors (List[AbstractUnbinnedFactor]): A list of factors that make up this component.

    Notes:
        - Factor names must be unique within a component.
    """

    def __init__(self, name: str, factors: List[AbstractUnbinnedFactor]):
        self.name = name
        self.factors = factors

        # Ensure factor names are unique
        factor_names = [factor.name for factor in factors]
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("Factor names must be unique")

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get the variables exposed by each factor in the model component.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping factor names to their exposed variables.
        """
        exposed = set()
        for factor in self.factors:
            exposed |= set(factor.exposed_parameters)
        return exposed

    @property
    def parameter_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the mapping of parameters to their corresponding factor names.

        Returns:
            Dict[str, str]: A dictionary mapping parameter names to their factor names.
        """
        return {factor.name: factor.parameter_mapping for factor in self.factors}

    @property
    def required_variables(self) -> Set[str]:
        """
        Collect all variables required by any factor in the model component.

        Returns:
            Set[str]: A set containing all required variables from all factors.
        """
        return {var for factor in self.factors for var in factor.required_variables}

    def evaluate(
        self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, float],
    ) -> Array:
        """
        Evaluate all factors in the model component in sequence, updating the output.

        Args:
            output (Array): Vector that will be modified by the evaluation.
            input_variables (Dict[str, Union[Array, float]]): Variables available as inputs to the factors.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated factors.

        Returns:
            Array: The modified output vector.
        """

        output = backend.array(1.0)

        for factor in self.factors:
            output *= factor.evaluate(input_variables, parameter_values)

        return output  # type: ignore

    def __repr__(self):
        """
        String representation of the ModelComponent object.

        Returns:
            str: A string representation of the model component.
        """
        lines = []
        lines.append(f"ModelComponent: {self.name}")
        lines.append(f"  Factors ({len(self.factors)}):")
        for factor in self.factors:
            # Use the factor's __repr__ method and indent it
            factor_repr = str(factor).replace('\n', '\n    ')
            lines.append(f"    {factor_repr}")
        lines.append(f"  Required variables: {sorted(self.required_variables)}")
        lines.append(f"  Exposed parameters: {sorted(self.exposed_parameters)}")
        return "\n".join(lines)

    def _repr_markdown_(self):
        """
        Markdown representation of the ModelComponent object.

        Returns:
            str: A markdown-formatted string representation of the model component.
        """
        return self.repr_markdown()

    def repr_markdown(
        self,
        indent_level: int = 0,
        bullet_style: str = "-",
        include_summary: bool = True,
    ) -> str:
        """
        Configurable markdown representation of the ModelComponent object.

        Args:
            indent_level (int): The level of indentation (each level adds 2 spaces). Default is 0.
            bullet_style (str): Style for bullet points ("-", "*", "+"). Default is "-".
            include_summary (bool): Whether to include required variables and exposed parameters summary. Default is True.

        Returns:
            str: A configurable markdown representation of the model component.
        """
        indent = "  " * indent_level
        sub_indent = "  " * (indent_level + 1)
        
        lines = []
        
        # Header with appropriate level based on indent
        if indent_level == 0:
            lines.append(f"#### ModelComponent: {self.name}")
        else:
            lines.append(f"{indent}{bullet_style} **ModelComponent:** {self.name}")
        
        # Factors section
        if indent_level == 0:
            lines.append(f"**Factors ({len(self.factors)}):**")
        else:
            lines.append(f"{sub_indent}{bullet_style} Factors ({len(self.factors)}):")
        
        for factor in self.factors:
            if hasattr(factor, 'repr_markdown'):
                # Use the configurable rendering with proper indentation level
                factor_md = factor.repr_markdown(
                    indent_level=indent_level + 1 if indent_level == 0 else indent_level + 2,
                    bullet_style=bullet_style,
                    include_type_in_name=True,
                )
                lines.append(factor_md)
            else:
                factor_indent = "" if indent_level == 0 else sub_indent
                lines.append(f"{factor_indent}{bullet_style} **{type(factor).__name__}** (`{factor.name}`) - No markdown representation available")
        
        if include_summary:
            lines.append("")
            if indent_level == 0:
                lines.append(f"**Required variables:** `{sorted(self.required_variables)}`")
                lines.append("")
                lines.append(f"**Exposed parameters:** `{sorted(self.exposed_parameters)}`")
            else:
                lines.append(f"{sub_indent}{bullet_style} Required variables: `{sorted(self.required_variables)}`")
                lines.append(f"{sub_indent}{bullet_style} Exposed parameters: `{sorted(self.exposed_parameters)}`")

        return "\n".join(lines)
