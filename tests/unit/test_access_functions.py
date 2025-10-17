# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL
# Copyright (c) 2025 Open Brain Institute

import os

import numpy as np
import pytest
import re
from numpy.testing import assert_array_equal
from bluepysnap import BluepySnapError, Circuit
from voxcell import VoxelData

from utils import TEST_DATA_DIR
import connectome_manipulator.access_functions as test_module


def test_get_node_ids():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = circuit.nodes[circuit.nodes.population_names[0]]

    # Check direct node set access
    for node_set in list(circuit.node_sets.content.keys()):
        ids = test_module.get_node_ids(nodes, node_set)
        assert_array_equal(ids, nodes.ids(node_set))

    # Check node access based on properties (layer)
    layers = list(np.unique(nodes.get(properties="layer")))
    for lay in layers:
        ids = test_module.get_node_ids(nodes, {"layer": lay})
        assert_array_equal(ids, nodes.ids({"layer": lay}))

    # Check combination of node set and properties (layer)
    for node_set in list(circuit.node_sets.content.keys()):
        for lay in layers:
            ids = test_module.get_node_ids(nodes, {"node_set": node_set, "layer": lay})
            ref_ids = np.intersect1d(nodes.ids(node_set), nodes.ids({"layer": lay}))
            assert_array_equal(ids, ref_ids)


def test_get_edges_population():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    popul_names = circuit.edges.population_names

    # Check selecting single (default) population
    edges = test_module.get_edges_population(circuit)
    assert edges is circuit.edges[popul_names[0]]

    # Check returned population name
    edges, pname = test_module.get_edges_population(circuit, return_popul_name=True)
    assert edges is circuit.edges[popul_names[0]]
    assert pname == popul_names[0]

    # Check selecting population by name (in case of single population)
    edges = test_module.get_edges_population(circuit, popul_names[0])
    assert edges is circuit.edges[popul_names[0]]

    # Check selecting population by name (in case of multiple populations)
    # (Not implemented: Would require test circuit with multiple populations)


def test_get_node_positions():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = circuit.nodes[circuit.nodes.population_names[0]]
    np.random.seed(0)
    node_ids = np.random.permutation(nodes.ids())
    ref_pos = nodes.positions(node_ids).to_numpy()

    # Case 1: No voxel map provided
    res = test_module.get_node_positions(nodes, node_ids, vox_map=None)
    assert_array_equal(ref_pos, res[0])  # raw_pos
    assert_array_equal(ref_pos, res[1])  # pos

    # Case 2: Voxel map provided
    vox_map = VoxelData.load_nrrd(os.path.join(TEST_DATA_DIR, "xy_map_lin.nrrd"))
    res = test_module.get_node_positions(nodes, node_ids, vox_map=vox_map)
    assert_array_equal(ref_pos, res[0])  # raw_pos
    assert_array_equal(vox_map.lookup(ref_pos), res[1])  # pos


def test_get_grouping():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = circuit.nodes["nodeA"]

    # Case 1: No node selection
    ## (a) No grouping
    group_by = None
    res_sel, res_val = test_module.get_grouping(nodes, None, group_by, skip_empty_groups=True)
    np.testing.assert_array_equal(res_sel, [None])
    np.testing.assert_array_equal(res_val, [None])

    ## (b) Invalid grouping
    group_by = "invalid"
    with pytest.raises(
        BluepySnapError, match=re.escape(f"Unknown node properties: ['{group_by}']")
    ):
        res_sel, res_val = test_module.get_grouping(nodes, None, group_by, skip_empty_groups=True)

    ## (c) Valid grouping
    group_by = "mtype"
    res_sel, res_val = test_module.get_grouping(nodes, None, group_by, skip_empty_groups=True)
    ref_mtypes = np.unique(nodes.get(properties="mtype"))
    np.testing.assert_array_equal(res_sel, [{"mtype": _mt} for _mt in ref_mtypes])
    np.testing.assert_array_equal(res_val, ref_mtypes)

    # Case 2: With node selection as str
    node_sel = "LayerA"
    ## (a) No grouping
    group_by = None
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=True)
    np.testing.assert_array_equal(res_sel, [None])
    np.testing.assert_array_equal(res_val, [None])

    ## (b) With grouping (with skip)
    group_by = "layer"
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=True)
    ref_layers = np.unique(nodes.get(node_sel, properties="layer"))
    np.testing.assert_array_equal(res_sel, [{"layer": _lay} for _lay in ref_layers])
    np.testing.assert_array_equal(res_val, ref_layers)

    ## (c) With grouping (w/o skip)
    group_by = "layer"
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=False)
    ref_layers = sorted(nodes.property_values("layer"))
    np.testing.assert_array_equal(res_sel, [{"layer": _lay} for _lay in ref_layers])
    np.testing.assert_array_equal(res_val, ref_layers)

    # Case 3: With node selection as dict
    node_sel = {"layer": "LA"}
    ## (a) No grouping
    group_by = None
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=True)
    np.testing.assert_array_equal(res_sel, [None])
    np.testing.assert_array_equal(res_val, [None])

    ## (b) Invalid grouping
    group_by = "invalid"
    with pytest.raises(
        BluepySnapError, match=re.escape(f"Unknown node properties: ['{group_by}']")
    ):
        res_sel, res_val = test_module.get_grouping(
            nodes, node_sel, group_by, skip_empty_groups=True
        )

    ## (c) Valid grouping (with skip)
    group_by = "mtype"
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=True)
    ref_mtypes = np.unique(nodes.get(node_sel, properties="mtype"))
    np.testing.assert_array_equal(res_sel, [{"mtype": _mt} for _mt in ref_mtypes])
    np.testing.assert_array_equal(res_val, ref_mtypes)

    ## (d) Valid grouping (w/o skip)
    group_by = "mtype"
    res_sel, res_val = test_module.get_grouping(nodes, node_sel, group_by, skip_empty_groups=False)
    ref_mtypes = sorted(nodes.property_values("mtype"))
    np.testing.assert_array_equal(res_sel, [{"mtype": _mt} for _mt in ref_mtypes])
    np.testing.assert_array_equal(res_val, ref_mtypes)
