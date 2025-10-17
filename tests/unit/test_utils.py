# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL
# Copyright (c) 2025 Open Brain Institute

import os
import re
import json
import pytest
from pathlib import Path
from cached_property import cached_property
from connectome_manipulator import utils as test_module
from bluepysnap.circuit import Circuit
from bluepysnap.nodes import NodePopulation
from unittest.mock import MagicMock

from utils import TEST_DATA_DIR


@pytest.mark.parametrize(
    "base_dir, path, expected",
    [
        ("/a/b", "$c", "$c"),
        ("/a/b", "", ""),
        ("/a/b", "/c/d", "/c/d"),
        ("/a/b", "/a/b/c", "$BASE_DIR/c"),
        ("/a/b", "/a/b/c/d", "$BASE_DIR/c/d"),
        ("/a/b/c", "/a/b/c/d", "$BASE_DIR/d"),
        ("/a/b", "./c", "$BASE_DIR/c"),
        ("/a/b", "c/d", "$BASE_DIR/c/d"),
    ],
)
def tested_reduce_path(base_dir, path, expected):
    result = test_module._reduce_path(path, Path(base_dir))
    assert result == expected


def test_reduce_config_paths__raises_relative_config_dir():
    expected = re.escape("Circuit config's directory is not absolute: .")
    with pytest.raises(ValueError, match=expected):
        test_module.reduce_config_paths(None, ".")


def test_reduce_config_paths__not_resolved_config():
    config_path = Path(TEST_DATA_DIR, "circuit_sonata.json")
    config = json.loads(config_path.read_bytes())
    contains = "A reduced config with absolute paths and no manifest must be provided."
    with pytest.raises(ValueError, match=contains):
        test_module.reduce_config_paths(config, "/")


def test_reduce_config_paths():
    config_path = Path(TEST_DATA_DIR, "circuit_sonata.json")

    reduced_config = Circuit(config_path).config

    res = test_module.reduce_config_paths(reduced_config, TEST_DATA_DIR)

    assert res["version"] == 2
    assert res["node_sets_file"] == "$BASE_DIR/node_sets.json"
    assert res["networks"] == {
        "nodes": [
            {
                "nodes_file": "$BASE_DIR/nodes.h5",
                "populations": {
                    "nodeA": {
                        "type": "biophysical",
                        "morphologies_dir": "$BASE_DIR/swc",
                        "biophysical_neuron_models_dir": "$BASE_DIR",
                    },
                },
            },
        ],
        "edges": [
            {
                "edges_file": "$BASE_DIR/edges.h5",
                "populations": {"nodeA__nodeA__chemical": {"type": "chemical"}},
            },
        ],
    }


def test_check_grouping():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    src_nodes = nodes[0]
    tgt_nodes = nodes[1]

    # Create mock source/target nodes without "layer" property
    src_nodes_mock = MagicMock()
    src_nodes_mock.property_names = {_p for _p in src_nodes.property_names if _p != "layer"}
    src_nodes_mock.get.return_value = src_nodes.get().drop("layer", axis=1)

    tgt_nodes_mock = MagicMock()
    tgt_nodes_mock.property_names = {_p for _p in tgt_nodes.property_names if _p != "layer"}
    tgt_nodes_mock.get.return_value = tgt_nodes.get().drop("layer", axis=1)

    # Case 1: No node populations provided
    ## (a) No grouping --> OK
    group_by = None
    res = test_module.check_grouping(group_by, src_nodes=None, tgt_nodes=None)
    assert res == (None, None)

    ## (b) No grouping (tuple) --> OK
    group_by = (None, None)
    res = test_module.check_grouping(group_by, src_nodes=None, tgt_nodes=None)
    assert res == (None, None)

    ## (c) No grouping (wrong tuple) --> ERROR
    group_by = (None, None, None)
    with pytest.raises(
        AssertionError,
        match=re.escape("'group_by' must be a tuple with two elements for source/target neurons"),
    ):
        res = test_module.check_grouping(group_by, src_nodes=None, tgt_nodes=None)

    ## (d) Grouping (str) --> OK
    group_by = "any"
    res = test_module.check_grouping(group_by, src_nodes=None, tgt_nodes=None)
    assert res == ("any", "any")

    ## (e) Grouping (tuple) --> OK
    group_by = ("any-A", "any-B")
    res = test_module.check_grouping(group_by, src_nodes=None, tgt_nodes=None)
    assert res == ("any-A", "any-B")

    # Case 2: Same source/target node populations provided
    ## (a) No grouping --> OK
    group_by = None
    res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)
    assert res == (None, None)

    ## (b) No grouping (tuple) --> OK
    group_by = (None, None)
    res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)
    assert res == (None, None)

    ## (c) Invalid grouping (str) --> ERROR
    group_by = "invalid"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by}' does not exist in either source or target node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)

    ## (d) Invalid source grouping (str) --> ERROR
    group_by = ("invalid", None)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by[0]}' does not exist in source node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)

    ## (e) Invalid target grouping (str) --> ERROR
    group_by = (None, "invalid")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by[1]}' does not exist in target node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)

    ## (f) Valid grouping (str) --> OK
    group_by = "layer"
    res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)
    assert res == ("layer", "layer")

    ## (g) Valid grouping (tuple) --> OK
    group_by = ("layer", "mtype")
    res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes)
    assert res == ("layer", "mtype")

    # Case 3: Different source/target node populations provided
    ## (a) Invalid grouping in src/tgt (str) --> ERROR
    group_by = "invalid"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by}' does not exist in either source or target node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes_mock)

    ## (b) Valid grouping in src only (str) --> OK (will ignore tgt)
    group_by = "layer"
    res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes_mock)
    assert res == ("layer", None)

    ## (c) Valid grouping in src only (tuple) --> ERROR (will not ignore tgt)
    group_by = ("layer", "layer")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by[1]}' does not exist in target node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes, tgt_nodes=tgt_nodes_mock)

    ## (d) Valid grouping in tgt only (str) --> OK (will ignore src)
    group_by = "layer"
    res = test_module.check_grouping(group_by, src_nodes=src_nodes_mock, tgt_nodes=tgt_nodes)
    assert res == (None, "layer")

    ## (e) Valid grouping in tgt only (tuple) --> ERROR (will not ignore src)
    group_by = ("layer", "layer")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Grouping property '{group_by[1]}' does not exist in source node population"
        ),
    ):
        res = test_module.check_grouping(group_by, src_nodes=src_nodes_mock, tgt_nodes=tgt_nodes)

    ## (f) Valid grouping in src/tgt (str) --> OK
    group_by = "mtype"
    res = test_module.check_grouping(group_by, src_nodes=src_nodes_mock, tgt_nodes=tgt_nodes)
    assert res == ("mtype", "mtype")

    ## (g) Valid grouping in src/tgt (tuple) --> OK
    group_by = ("mtype", "layer")
    res = test_module.check_grouping(group_by, src_nodes=src_nodes_mock, tgt_nodes=tgt_nodes)
    assert res == ("mtype", "layer")
