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
            parameter_values (Dict[str, float]): Variables exposed by previously evaluated expectations.

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
                model_repr = str(model).replace("\n", "\n    ")
                lines.append(f"    {model_repr}")

            # Binning information using its __repr__ method
            lines.append("  Binning:")
            binning_repr = str(expectation.binning).replace("\n", "\n    ")
            lines.append(f"    {binning_repr}")

            # Binned factors with detailed information
            if expectation.binned_factors:
                lines.append(f"  Binned factors ({len(expectation.binned_factors)}):")
                for factor in expectation.binned_factors:
                    factor_repr = str(factor).replace("\n", "\n    ")
                    lines.append(f"    {factor_repr}")

            # Variables and parameters for this expectation
            lines.append(
                f"  Required variables: {sorted(expectation.required_variables)}"
            )
            lines.append(
                f"  Exposed parameters: {sorted(expectation.exposed_parameters)}"
            )
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
            lines.append(
                f"{sub_indent}{bullet_style} {len(self.expectations)} expectations configured"
            )

        lines.append("")

        # Overall summary
        if include_summary:
            if indent_level == 0:
                lines.append("## Summary")
                lines.append("| Aspect | Details |")
                lines.append("|--------|---------|")
                lines.append(
                    f"| Required variables | `{', '.join(sorted(self.required_variables))}` |"
                )
                lines.append(
                    f"| Exposed parameters | `{', '.join(sorted(self.exposed_parameters))}` |"
                )
            else:
                lines.append(f"{sub_indent}{bullet_style} Summary:")
                summary_indent = "  " * (indent_level + 2)
                lines.append(
                    f"{summary_indent}{bullet_style} Required variables: `{', '.join(sorted(self.required_variables))}`"
                )
                lines.append(
                    f"{summary_indent}{bullet_style} Exposed parameters: `{', '.join(sorted(self.exposed_parameters))}`"
                )
            lines.append("")

        # Detailed expectation information
        if show_expectation_details:
            for exp_name, expectation in self.expectations.items():
                exp_md = expectation.repr_markdown(
                    indent_level=indent_level + 1
                    if indent_level == 0
                    else indent_level + 2,
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
                exp_indent = (
                    sub_indent if indent_level == 0 else "  " * (indent_level + 2)
                )
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

    def render_graph(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 3000,
        font_size: int = 10,
        show_datasets: bool = True,
        show_parameters: bool = True,
        layout: str = "spring",
        engine: str = "networkx",
    ):
        """
        Render the analysis object and its subcomponents as a graph visualization.
        Uses NetworkX for graph creation and either NetworkX/matplotlib or Graphviz for rendering.

        Args:
            figsize: Figure size as (width, height) tuple
            node_size: Base size of nodes in the graph (used when auto_size_nodes=False and engine='networkx')
            font_size: Font size for node labels
            show_datasets: Whether to show dataset nodes
            show_parameters: Whether to show parameter nodes
            layout: Layout algorithm ('spring', 'hierarchical', 'circular' for NetworkX; 'dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi' for Graphviz)
            engine: Rendering engine ('networkx' or 'graphviz')

        Returns:
            matplotlib figure object (NetworkX) or Graphviz AGraph object (Graphviz)
        """
        # Create the NetworkX graph structure (shared by both engines)
        G = self._create_networkx_graph(
            show_datasets=show_datasets, show_parameters=show_parameters
        )

        if engine.lower() == "graphviz":
            return self._render_with_graphviz(
                G,
                figsize=figsize,
                font_size=font_size,
                layout=layout,
            )
        else:
            return self._render_with_networkx(
                G,
                figsize=figsize,
                node_size=node_size,
                font_size=font_size,
                layout=layout,
                show_parameters=show_parameters,
            )

    def _create_networkx_graph(
        self, show_datasets: bool = True, show_parameters: bool = True
    ):
        """
        Create the NetworkX graph structure representing the analysis.

        Args:
            show_datasets: Whether to include dataset nodes
            show_parameters: Whether to include parameter nodes

        Returns:
            nx.DiGraph: The created NetworkX directed graph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Graph creation requires networkx. Install with: pip install networkx"
            )

        # Create directed graph
        G = nx.DiGraph()

        # Add analysis root node
        G.add_node("Analysis", node_type="analysis", label="Analysis")

        # Process each expectation
        for exp_name, expectation in self.expectations.items():
            # Add expectation node
            exp_node = f"Expectation_{exp_name}"
            G.add_node(
                exp_node, node_type="expectation", label=f"Expectation\n{exp_name}"
            )
            G.add_edge("Analysis", exp_node)

            # Add binning node
            binning_node = f"Binning_{exp_name}"
            binning_info = expectation.binning
            binning_label = f"Binning\n{type(binning_info).__name__}"
            # Try to get variables information if available
            if hasattr(binning_info, "required_variables"):
                vars_list = list(binning_info.required_variables)[:3]  # Limit display
                if vars_list:
                    binning_label += f"\n{', '.join(vars_list)}"
                    if len(binning_info.required_variables) > 3:
                        binning_label += "..."
            G.add_node(binning_node, node_type="binning", label=binning_label)
            G.add_edge(exp_node, binning_node)

            # Process dataset-model pairs
            for ds_key, model in expectation.dskey_model_pairs:
                # Add dataset
                dataset_node = None
                if show_datasets:
                    dataset_node = f"Dataset_{ds_key}"
                    G.add_node(
                        dataset_node, node_type="dataset", label=f"Dataset\n{ds_key}"
                    )

                # Add model
                model_node = f"Model_{model.name}_{exp_name}"
                G.add_node(model_node, node_type="model", label=f"Model\n{model.name}")
                G.add_edge(exp_node, model_node)

                if show_datasets and dataset_node is not None:
                    G.add_edge(dataset_node, model_node)

                # Process model components
                for i, component in enumerate(model.components):
                    comp_node = f"Component_{component.name}_{model.name}_{exp_name}"
                    baseline_weight = (
                        model.baseline_weights[i]
                        if i < len(model.baseline_weights)
                        else "unknown"
                    )
                    comp_label = (
                        f"Component\n{component.name}\n(weight: {baseline_weight})"
                    )
                    G.add_node(comp_node, node_type="component", label=comp_label)
                    G.add_edge(model_node, comp_node)

                    # Process factors within component
                    for factor in component.factors:
                        factor_node = f"Factor_{factor.name}_{component.name}_{model.name}_{exp_name}"
                        factor_label = (
                            f"Factor\n{factor.name}\n({type(factor).__name__})"
                        )
                        G.add_node(factor_node, node_type="factor", label=factor_label)
                        G.add_edge(comp_node, factor_node)

        # Add parameter nodes if requested
        if show_parameters:
            # Create a mapping from factor nodes to their factor objects for easier lookup
            factor_node_to_object = {}

            for exp_name, expectation in self.expectations.items():
                for ds_key, model in expectation.dskey_model_pairs:
                    for component in model.components:
                        for factor in component.factors:
                            factor_node = f"Factor_{factor.name}_{component.name}_{model.name}_{exp_name}"
                            factor_node_to_object[factor_node] = factor

            for factor_node, factor_obj in factor_node_to_object.items():
                params = factor_obj.exposed_parameters
                for param in params:
                    param_node = f"Parameter_{param}"
                    G.add_node(
                        param_node, node_type="parameter", label=f"Parameter\n{param}"
                    )
                    G.add_edge(param_node, factor_node)

        return G

    def _render_with_networkx(
        self, G, figsize, node_size, font_size, layout, show_parameters
    ):
        """Render using NetworkX and matplotlib."""
        try:
            import matplotlib.lines as mlines
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                f"NetworkX rendering requires matplotlib and networkx: {e}"
            )

        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == "hierarchical":
            # Create hierarchical layout with optimized parameter ordering
            pos = {}
            levels = {
                "analysis": 0,
                "expectation": 1,
                "model": 2,
                "component": 3,
                "factor": 4,
                "binning": 1,
                "dataset": 2,
                "parameter": 5,
            }

            # Group nodes by level
            level_nodes = {}
            for node, data in G.nodes(data=True):
                level = levels.get(data.get("node_type", "unknown"), 0)
                if level not in level_nodes:
                    level_nodes[level] = []
                level_nodes[level].append(node)

            # Position nodes with optimized spacing
            for level, nodes in level_nodes.items():
                if (
                    level == 4
                ):  # Factors level - use wider spacing for better parameter alignment
                    spacing = 3.0 if show_parameters else 2.0
                elif level == 5:  # Parameters level - use consistent spacing
                    spacing = 2.5
                else:
                    spacing = 2.0

                for i, node in enumerate(nodes):
                    x = (i - len(nodes) / 2) * spacing
                    y = -level * 2.5  # Increase vertical spacing slightly
                    pos[node] = (x, y)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:  # spring layout (default)
            pos = nx.spring_layout(G, k=3, iterations=50)

        # Define colors for different node types
        colors = {
            "analysis": "#FF6B6B",  # Red
            "expectation": "#4ECDC4",  # Teal
            "model": "#45B7D1",  # Blue
            "component": "#96CEB4",  # Light Green
            "factor": "#FCEA2B",  # Yellow
            "binning": "#FF9999",  # Light Red
            "dataset": "#DDA0DD",  # Plum
            "parameter": "#FFB347",  # Orange
        }

        # Draw nodes by type with adaptive sizing
        for node_type, color in colors.items():
            nodes_of_type = [
                node
                for node, data in G.nodes(data=True)
                if data.get("node_type") == node_type
            ]
            if nodes_of_type:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=nodes_of_type,
                    node_color=color,
                    node_size=node_size,  # type: ignore
                    alpha=0.8,
                    ax=ax,
                    node_shape="o",  # Use circle shape which is valid
                )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.6, ax=ax
        )

        # Draw labels
        labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=font_size, font_weight="bold", ax=ax
        )

        # Create legend
        legend_elements = [
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=node_type.capitalize(),
            )
            for node_type, color in colors.items()
            if any(data.get("node_type") == node_type for _, data in G.nodes(data=True))
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_title("Analysis Structure Graph", fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        return fig

    def _render_with_graphviz(self, G, figsize, font_size, layout):
        """Render using Graphviz via NetworkX conversion."""
        try:
            from networkx.drawing.nx_agraph import to_agraph
        except ImportError:
            raise ImportError(
                "Graphviz rendering requires pygraphviz. Install with: pip install pygraphviz"
            )

        # Define node styles for different types
        node_styles = {
            "analysis": {
                "shape": "box",
                "style": "filled",
                "fillcolor": "#FF6B6B",
                "fontcolor": "white",
            },
            "expectation": {
                "shape": "box",
                "style": "filled",
                "fillcolor": "#4ECDC4",
                "fontcolor": "black",
            },
            "model": {
                "shape": "box",
                "style": "filled",
                "fillcolor": "#45B7D1",
                "fontcolor": "white",
            },
            "component": {
                "shape": "box",
                "style": "filled",
                "fillcolor": "#96CEB4",
                "fontcolor": "black",
            },
            "factor": {
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "#FCEA2B",
                "fontcolor": "black",
            },
            "binning": {
                "shape": "diamond",
                "style": "filled",
                "fillcolor": "#FF9999",
                "fontcolor": "black",
            },
            "dataset": {
                "shape": "folder",
                "style": "filled",
                "fillcolor": "#DDA0DD",
                "fontcolor": "black",
            },
            "parameter": {
                "shape": "circle",
                "style": "filled",
                "fillcolor": "#FFB347",
                "fontcolor": "black",
            },
        }

        # Apply visual attributes to NetworkX graph for Graphviz
        for node, data in G.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            if node_type in node_styles:
                # Apply Graphviz attributes to the node
                for attr, value in node_styles[node_type].items():
                    G.nodes[node][attr] = value

            # Set font size and name
            G.nodes[node]["fontsize"] = str(font_size)
            G.nodes[node]["fontname"] = "Arial"

            # Escape newlines in labels for Graphviz
            if "label" in data:
                G.nodes[node]["label"] = data["label"].replace("\n", "\\n")

        # Set edge attributes
        for edge in G.edges():
            G.edges[edge]["fontsize"] = str(max(8, font_size - 2))

        # Convert NetworkX graph to AGraph
        A = to_agraph(G)

        # Set graph attributes
        A.graph_attr.update(rankdir="TB", size=f"{figsize[0]},{figsize[1]}!")
        A.node_attr.update(fontsize=str(font_size), fontname="Arial")
        A.edge_attr.update(fontsize=str(max(8, font_size - 2)))

        # Apply layout
        A.layout(prog=layout)

        return A
