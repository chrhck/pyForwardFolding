from typing import Dict, List, Set, Tuple, Union

from .backend import Array, backend
from .model_component import ModelComponent


class Model:
    """
    A forward-folding model consisting of multiple components, each with an associated baseline weight.

    Args:
        name (str): The name of the model.
        components (List[ModelComponent]): The components that make up the model.
        baseline_weights (List[str]): Symbols (as strings) representing the baseline weight for each component.

    Notes:
        - Component names must be unique within a model.
    """

    def __init__(
        self, name: str, components: List[ModelComponent], baseline_weights: List[str]
    ):
        self.name = name
        self.components = components
        self.baseline_weights = baseline_weights

        # Ensure component names are unique
        component_names = [component.name for component in components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("Model components must have unique names")

    @classmethod
    def from_pairs(
        cls, name: str, components: List[Tuple[str, ModelComponent]]
    ) -> "Model":
        """
        Alternate constructor to create a Model from a list of (baseline_weight, component) pairs.

        Args:
            name (str): The name of the model.
            components (List[Tuple[str, ModelComponent]]): A list of (baseline_weight, component) pairs.

        Returns:
            Model: A new Model instance.
        """
        baseline_weights = [pair[0] for pair in components]
        component_list = [pair[1] for pair in components]
        return cls(name, component_list, baseline_weights)

    @property
    def required_variables(self) -> Set[str]:
        """
        Get all variables required by any component in the model.

        Returns:
            Set[str]: A set containing all required variables.
        """
        return {
            var for component in self.components for var in component.required_variables
        }

    @property
    def exposed_parameters(self) -> Set[str]:
        """
        Get all parameters exposed by the model's components.

        Returns:
            Dict[str, Dict[str, str]]: A merged dictionary of all exposed parameters from all components.
        """

        return {
            par for component in self.components for par in component.exposed_parameters
        }

    @property
    def parameter_mapping(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Get the mapping of model parameters to their respective components.

        Returns:
            Dict[str, Dict[str, Dict[str, str]]]: A dictionary mapping component names to their parameter names.
        """
        return {
            component.name: component.parameter_mapping for component in self.components
        }

    def evaluate(
        self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, float],
    ) -> Array:
        """
        Evaluate the model by computing the sum of all components.

        Args:
            input_variables (Dict[str, Union[Array, float]]): Input variables for model evaluation.
            parameter_values (Dict[str, Union[Array, float]]): Variables exposed by previously evaluated components.

        Returns:
            Array: The modified `output` containing the sum of all components.

        Raises:
            ValueError: If any baseline weight is not found in the input variables.
        """

        output = backend.array(0.0)
        for component, baseline_weight in zip(self.components, self.baseline_weights):
            # Get the baseline weight value
            baseline_weight_value = input_variables.get(baseline_weight, None)
            if baseline_weight_value is None:
                raise ValueError(
                    f"Baseline weight '{baseline_weight}' not found in input variables"
                )

            # Initialize the component buffer with the baseline weight value
            comp_eval = baseline_weight_value

            # Evaluate the component
            comp_eval *= component.evaluate(input_variables, parameter_values)

            # Accumulate the result into the output
            output += comp_eval

        return output

    def evaluate_per_component(
        self,
        input_variables: Dict[str, Union[Array, float]],
        parameter_values: Dict[str, float],
    ) -> Dict[str, Union[Array, float]]:
        """
        Evaluate each component of the model individually.
        Args:
            input_variables (Dict[str, Union[Array, float]]): Input variables for model evaluation.
            parameter_values (Dict[str, float]): Variables exposed by previously evaluated components.
        Returns:
            Dict[str, Union[Array, float]]: A dictionary containing evaluation results for each component.
        Raises:
            ValueError: If any baseline weight is not found in the input variables.
        """

        # Ensure all input variables have the same length
        input_var_lengths = [
            len(value) if isinstance(value, Array) else 1
            for value in input_variables.values()
        ]
        if not all(length == input_var_lengths[0] for length in input_var_lengths):
            raise ValueError("All input variables must have the same length")
        output: Dict[str, Union[Array, float]] = {}
        for component, baseline_weight in zip(self.components, self.baseline_weights):
            # Get the baseline weight value
            baseline_weight_value = input_variables.get(baseline_weight, None)
            if baseline_weight_value is None:
                raise ValueError(
                    f"Baseline weight '{baseline_weight}' not found in input variables"
                )

            # Initialize the component buffer with the baseline weight value
            comp_eval = baseline_weight_value

            # Evaluate the component
            comp_eval *= component.evaluate(input_variables, parameter_values)

            # Accumulate the result into the output
            output[component.name] = comp_eval

        return output

    def __repr__(self):
        """
        String representation of the Model object.

        Returns:
            str: A string representation of the model.
        """
        lines = []
        lines.append(f"Model: {self.name}")
        lines.append(f"  Components ({len(self.components)}):")
        for comp, weight in zip(self.components, self.baseline_weights):
            lines.append(f"    • {comp.name} (weight: {weight})")
            # Indent the component representation
            comp_repr = str(comp).replace('\n', '\n      ')
            lines.append(f"      {comp_repr}")
        lines.append(f"  Required variables: {sorted(self.required_variables)}")
        lines.append(f"  Exposed parameters: {sorted(self.exposed_parameters)}")
        return "\n".join(lines)

    def _repr_markdown_(self):
        """
        Markdown representation of the Model object.

        Returns:
            str: A markdown-formatted string representation of the model.
        """
        return self.repr_markdown()

    def repr_markdown(
        self,
        indent_level: int = 0,
        bullet_style: str = "-",
        include_summary: bool = True,
        show_component_details: bool = True,
    ) -> str:
        """
        Configurable markdown representation of the Model object.

        Args:
            indent_level (int): The level of indentation (each level adds 2 spaces). Default is 0.
            bullet_style (str): Style for bullet points ("-", "*", "+"). Default is "-".
            include_summary (bool): Whether to include required variables and exposed parameters summary. Default is True.
            show_component_details (bool): Whether to show detailed component markdown. Default is True.

        Returns:
            str: A configurable markdown representation of the model.
        """
        indent = "  " * indent_level
        sub_indent = "  " * (indent_level + 1)
        
        lines = []
        
        # Header with appropriate level based on indent
        if indent_level == 0:
            lines.append(f"### Model: {self.name}")
        else:
            lines.append(f"{indent}{bullet_style} **Model:** {self.name}")
        
        if include_summary:
            lines.append("")
            if indent_level == 0:
                lines.append(f"**Required variables:** `{sorted(self.required_variables)}`\n")
                lines.append(f"**Exposed parameters:** `{sorted(self.exposed_parameters)}`")
            else:
                lines.append(f"{sub_indent}{bullet_style} Required variables: `{sorted(self.required_variables)}`")
                lines.append(f"{sub_indent}{bullet_style} Exposed parameters: `{sorted(self.exposed_parameters)}`")



        # Components section
        if indent_level == 0:
            lines.append(f"**Components ({len(self.components)}):**")
        else:
            lines.append(f"{sub_indent}{bullet_style} Components ({len(self.components)}):")
        
        for comp, weight in zip(self.components, self.baseline_weights):
            comp_indent = "" if indent_level == 0 else sub_indent
            lines.append(f"{comp_indent}{bullet_style} **{comp.name}** (weight: {weight})")
            
            if show_component_details and hasattr(comp, 'repr_markdown'):
                # Use the configurable rendering with proper indentation level
                comp_md = comp.repr_markdown(
                    indent_level=indent_level + 1 if indent_level == 0 else indent_level + 2,
                    bullet_style=bullet_style,
                    include_summary=include_summary,
                )
                lines.append(comp_md)
            elif show_component_details:
                fallback_indent = sub_indent if indent_level == 0 else "  " * (indent_level + 2)
                lines.append(f"{fallback_indent}(No markdown representation available for {comp.name})")
        
       
        return "\n".join(lines)
