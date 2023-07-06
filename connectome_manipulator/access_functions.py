"""Collection of function for flexible nodes/edges access, to be used by model building and manipulation operations"""
from pathlib import Path

import numpy as np
import pandas as pd

import libsonata
from bluepysnap.sonata_constants import Node, DYNAMICS_PREFIX
from bluepysnap.utils import add_dynamic_prefix
from bluepysnap.utils import euler2mat, quaternion2mat

from connectome_manipulator import log


def property_names(nodes):
    """Get all property names for a population"""
    population = nodes._population  # pylint: disable=protected-access
    return set(population.attribute_names) | set(
        add_dynamic_prefix(population.dynamics_attribute_names)
    )


def get_nodes(nodes, selection=None):
    """Get a pandas table of all nodes and their properties, optionally narrowed by a selection"""
    population = nodes._population  # pylint: disable=protected-access
    categoricals = population.enumeration_names

    if selection is None:
        selection = nodes.select_all()
    result = pd.DataFrame(index=selection.flatten())

    for attr in sorted(population.attribute_names):
        if attr in categoricals:
            enumeration = np.asarray(population.get_enumeration(attr, selection))
            values = np.asarray(population.enumeration_values(attr))
            # if the size of `values` is large enough compared to `enumeration`, not using
            # categorical reduces the memory usage.
            if values.shape[0] < 0.5 * enumeration.shape[0]:
                result[attr] = pd.Categorical.from_codes(enumeration, categories=values)
            else:
                result[attr] = values[enumeration]
        else:
            result[attr] = population.get_attribute(attr, selection)
    for attr in sorted(add_dynamic_prefix(population.dynamics_attribute_names)):
        result[attr] = population.get_dynamics_attribute(attr.split(DYNAMICS_PREFIX)[1], selection)
    return result


def get_morphology_paths(nodes, selection, morpho_helper, extension="swc"):
    """Return paths to morphology files corresponding to `selection`.

    Args:
        nodes: a bluepysnap node set
        selection (libsonata.Selection): a selection of nodes
        morpho_helper: a bluepysnap MorphoHelper instance
        extension (str): expected filetype extension of the morph file.
    """
    morpho_dir = morpho_helper._get_morph_dir(extension)  # pylint: disable=protected-access
    result = get_nodes(nodes, selection)
    return [Path(morpho_dir, f"{name}.{extension}") for name in result[Node.MORPHOLOGY]]


def orientations(nodes, node_sel=None):
    """Node orientation(s) as a list of numpy arrays.

    Args:
        nodes: the node set for which we want to get the rotation matrices
        node_sel: (optional) a libsonata Selection to narrow our selection

    Returns:
        numpy.ndarray:
            A list of 3x3 rotation matrices for the given node set and selection.
    """
    # need to keep this quaternion ordering for quaternion2mat (expects w, x, y , z)
    props = np.array(
        [Node.ORIENTATION_W, Node.ORIENTATION_X, Node.ORIENTATION_Y, Node.ORIENTATION_Z]
    )
    props_mask = np.isin(props, list(property_names(nodes)))
    orientation_count = np.count_nonzero(props_mask)
    if orientation_count == 4:
        trans = quaternion2mat
    elif orientation_count in [1, 2, 3]:
        raise ValueError(
            "Missing orientation fields. Should be 4 quaternions or euler angles or nothing"
        )
    else:
        # need to keep this rotation_angle ordering for euler2mat (expects z, y, x)
        props = np.array(
            [
                Node.ROTATION_ANGLE_Z,
                Node.ROTATION_ANGLE_Y,
                Node.ROTATION_ANGLE_X,
            ]
        )
        props_mask = np.isin(props, list(property_names(nodes)))
        trans = euler2mat
    result = get_nodes(nodes, node_sel)
    if props[props_mask].size:
        result = result[props[props_mask]]

    def _get_values(prop):
        """Retrieve prop from the result Dataframe/Series."""
        if isinstance(result, pd.Series):
            return [result.get(prop, 0)]
        return result.get(prop, np.zeros((result.shape[0],)))

    args = [_get_values(prop) for prop in props]
    return trans(*args)


def get_enumeration_list(pop, column):
    """Takes a node population and column name and returns a list to values."""
    raw_pop = pop._population  # pylint: disable=protected-access
    if column in raw_pop.enumeration_names:
        return raw_pop.enumeration_values(column)
    return sorted(np.unique(raw_pop.get_attribute(column, raw_pop.select_all())))


def get_enumeration_map(pop, column):
    """Takes a node population and column name and returns a dictionary that maps values to indices."""
    raw_pop = pop._population  # pylint: disable=protected-access
    if column in raw_pop.enumeration_names:
        return {key: idx for idx, key in enumerate(raw_pop.enumeration_values(column))}
    return {
        key: idx
        for idx, key in enumerate(
            sorted(np.unique(raw_pop.get_attribute(column, raw_pop.select_all())))
        )
    }


def get_attribute(pop, column, ids):
    """Get the attribute values for `column` from population `pop` for node IDs `ids`."""
    raw_pop = pop._population  # pylint: disable=protected-access
    return raw_pop.get_attribute(column, libsonata.Selection(ids))


def get_enumeration(pop, column, ids=None):
    """Get the raw enumeration values for `column` from population `pop` for node IDs `ids`."""
    raw_pop = pop._population  # pylint: disable=protected-access
    if ids is None:
        ids = raw_pop.select_all()
    else:
        ids = libsonata.Selection(ids)
    if column in raw_pop.enumeration_names:
        return raw_pop.get_enumeration(column, ids)
    mapping = get_enumeration_map(pop, column)
    return np.array([mapping[v] for v in raw_pop.get_attribute(column, ids)])


def get_node_ids(nodes, sel_spec, split_ids=None):
    """Returns list of selected node IDs of given nodes population.

    nodes ... NodePopulation
    sel_spec ... Node selection specifier, as accepted by nodes.ids(group=sel_spec).
                 In addition, if sel_spec is a dict, 'node_set': '<node_set_name>'
                 can be specified in combination with other selection properties.
    split_ids ... Node IDs to filter the selection by, either as an array or a
                  libsonata.Selection
    """
    pop = nodes._population  # pylint: disable=protected-access
    enumeration_names = pop.enumeration_names
    if split_ids is None:
        sel_ids = pop.select_all()
    else:
        sel_ids = libsonata.Selection(split_ids)
    if isinstance(sel_spec, dict):
        sel_group = sel_spec.copy()
        node_set = sel_group.pop("node_set", None)

        selection = None
        for sel_k, sel_v in sel_group.items():
            if sel_k in enumeration_names:
                sel_idx = pop.enumeration_values(sel_k).index(sel_v)
                sel_prop = pop.get_enumeration(sel_k, sel_ids) == sel_idx
            else:
                sel_prop = pop.get_attribute(sel_k, sel_ids) == sel_v
            if selection is None:
                selection = sel_prop
            else:
                selection &= sel_prop
        # selection is not of all nodes (starting with node id 0), but a generic subset specified by sel_ids
        if len(sel_ids.ranges) == 1:
            # First turn selection array into an index array then
            # because we filtered contiguous ids with a simple offset, just shift by the first node id
            gids = np.nonzero(selection)[0] + sel_ids.ranges[0][0]
        else:
            # for more complex cases, fully resolve the node-preselection
            gids = sel_ids.flatten().astype(np.int64)[selection]

        if node_set is not None:
            log.log_assert(isinstance(node_set, str), "Node set must be a string!")
            gids = np.intersect1d(gids, nodes.ids(node_set))
    else:
        gids = nodes.ids(sel_spec)
        if split_ids is not None:
            gids = np.intersect1d(gids, sel_ids.flatten().astype(np.int64))

    return gids


def get_nodes_population(circuit, popul_name=None):
    """Select default edge population. Optionally, the population name can be specified."""
    log.log_assert(len(circuit.nodes.population_names) > 0, "No node population found!")
    if popul_name is None:
        if len(circuit.nodes.population_names) == 1:
            popul_name = circuit.nodes.population_names[0]  # Select the only existing population
        else:
            popul_name = "All"  # Use default name
            log.warning(
                f'Multiple node populations found - Trying to load "{popul_name}" population!'
            )
    log.log_assert(
        popul_name in circuit.nodes.population_names,
        f'Population "{popul_name}" not found in edges file!',
    )
    nodes = circuit.nodes[popul_name]

    return nodes


def get_edges_population(circuit, popul_name=None):
    """Select default edge population. Optionally, the population name can be specified."""
    #     log.log_assert(len(circuit.edges.population_names) == 1, 'Only a single edge population per file supported!')
    #     edges = circuit.edges[circuit.edges.population_names[0]]
    log.log_assert(len(circuit.edges.population_names) > 0, "No edge population found!")
    if popul_name is None:
        if len(circuit.edges.population_names) == 1:
            popul_name = circuit.edges.population_names[0]  # Select the only existing population
        else:
            popul_name = "default"  # Use default name
            log.warning(
                f'Multiple edges populations found - Trying to load "{popul_name}" population!'
            )
    log.log_assert(
        popul_name in circuit.edges.population_names,
        f'Population "{popul_name}" not found in edges file!',
    )
    edges = circuit.edges[popul_name]

    return edges
