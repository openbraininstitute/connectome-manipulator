# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL
# Copyright (c) 2025 Open Brain Institute

import os

import numpy as np
import pandas as pd
import pytest
import re
from bluepysnap import Circuit
from numpy.testing import assert_array_equal, assert_allclose

from utils import TEST_DATA_DIR

import connectome_manipulator.connectome_comparison.properties as test_module


def _get_props(edges_table, src_ids, tgt_ids, nodes, group_by, prop_name):
    """Extract connectivity matrices from edges table"""
    conns = np.unique(edges_table[["@source_node", "@target_node"]], axis=0)
    prop_mat_sum = np.zeros((len(src_ids), len(tgt_ids)), dtype=float) * np.nan
    prop_mat_cnt = np.zeros((len(src_ids), len(tgt_ids)), dtype=int)
    for _s, _t in conns:
        edge_sel = np.logical_and(
            edges_table["@source_node"] == _s, edges_table["@target_node"] == _t
        )
        prop_mat_sum[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = edges_table.loc[
            edge_sel, prop_name
        ].sum()
        prop_mat_cnt[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = np.sum(edge_sel)

    if not isinstance(group_by, tuple):
        group_by = (group_by, group_by)
    if group_by[0] is None:
        src_grp = np.zeros_like(src_ids)
    else:
        src_grp = nodes[0].get(src_ids, properties=group_by[0]).values
    if group_by[1] is None:
        tgt_grp = np.zeros_like(tgt_ids)
    else:
        tgt_grp = nodes[1].get(tgt_ids, properties=group_by[1]).values

    df_props_sum = pd.DataFrame(prop_mat_sum, columns=tgt_grp)
    df_props_sum["src"] = src_grp
    df_props_sum = df_props_sum.melt("src", var_name="tgt", value_name="data")
    df_props_sum = df_props_sum.groupby(["src", "tgt"]).sum()

    df_props_cnt = pd.DataFrame(prop_mat_cnt, columns=tgt_grp)
    df_props_cnt["src"] = src_grp
    df_props_cnt = df_props_cnt.melt("src", var_name="tgt", value_name="data")
    df_props_cnt = df_props_cnt.groupby(["src", "tgt"]).sum()

    df_props = df_props_sum / df_props_cnt

    return df_props.unstack()


def _check_props(res, df_props, group_by, prop_name):
    for _key in [prop_name, "common"]:
        assert _key in res, f'ERROR: Results key "{_key}" missing!'
    for _key in ["src_group_values", "tgt_group_values"]:
        assert _key in res["common"], f'ERROR: Results key "{_key}" in "common" missing!'
    assert "data" in res[prop_name], f'ERROR: Results key "data" in "{prop_name}" missing!'

    if not isinstance(group_by, tuple):
        group_by = (group_by, group_by)
    if group_by[0] is None:
        assert_array_equal(
            res["common"]["src_group_values"], [None], "ERROR: Source group mismatch!"
        )
    else:
        assert_array_equal(
            res["common"]["src_group_values"],
            df_props.index.to_numpy(),
            "ERROR: Source group mismatch!",
        )
    if group_by[1] is None:
        assert_array_equal(
            res["common"]["tgt_group_values"], [None], "ERROR: Target group mismatch!"
        )
    else:
        assert_array_equal(
            res["common"]["tgt_group_values"],
            df_props.columns.get_level_values(1).to_numpy(),
            "ERROR: Target group mismatch!",
        )
    assert_allclose(
        res[prop_name]["data"],
        df_props.to_numpy(),
        rtol=1e-06,
        err_msg=f'ERROR: "{prop_name}" mismatch!',
    )


def _check_empty(res, src_ids, tgt_ids, nodes, edges, group_by, skip):
    nsrc = len(nodes[0].property_values(group_by))
    ntgt = len(nodes[1].property_values(group_by))
    res_keys = list(edges.property_names)
    for key in res_keys:
        if len(src_ids) == 0:  # No src selection
            if skip:
                assert res[key]["data"].shape[0] == 0
            else:
                assert res[key]["data"].shape[0] == nsrc
        else:
            assert res[key]["data"].shape[0] > 0

        if len(tgt_ids) == 0:  # No tgt selection
            if skip:
                assert res[key]["data"].shape[1] == 0
            else:
                assert res[key]["data"].shape[1] == ntgt
        else:
            assert res[key]["data"].shape[1] > 0


def test_properties():
    circuit = Circuit(os.path.join(TEST_DATA_DIR, "circuit_sonata.json"))
    nodes = [circuit.nodes["nodeA"]] * 2  # Src/tgt populations
    edges = circuit.edges["nodeA__nodeA__chemical"]
    edges_table = edges.afferent_edges(nodes[1].ids(), properties=edges.property_names)

    # Case 1: Invalid inputs
    ## (a) Invalid population name
    popul_name = "INVALID_POPULATION_NAME"
    with pytest.raises(
        AssertionError, match=re.escape(f'Population "{popul_name}" not found in edges file')
    ):
        res = test_module.compute(
            circuit,
            sel_src=None,
            sel_dest=None,
            group_by=None,
            skip_empty_groups=False,
            edges_popul_name=popul_name,
        )

    ## (b) Invalid group-by
    popul_name = "nodeA__nodeA__chemical"
    with pytest.raises(
        AssertionError,
        match=re.escape("'group_by' must be a tuple with two elements for source/target neurons"),
    ):
        res = test_module.compute(
            circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=("mtype",)
        )
    with pytest.raises(
        AssertionError, match=re.escape("Source 'group_by' must be a string (or None)")
    ):
        res = test_module.compute(
            circuit,
            sel_src=None,
            sel_dest=None,
            edges_popul_name=popul_name,
            group_by=(123, "mtype"),
        )
    with pytest.raises(
        AssertionError, match=re.escape("Target 'group_by' must be a string (or None)")
    ):
        res = test_module.compute(
            circuit,
            sel_src=None,
            sel_dest=None,
            edges_popul_name=popul_name,
            group_by=("mtype", 123),
        )

    # Case 2: Full circuit
    ## (a) W/o group-by
    group_by = None
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    for eprop in edges.property_names:
        df_props = _get_props(edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by, eprop)
        _check_props(res, df_props, group_by, eprop)

    ## (b) W/ group-by
    group_by = "layer"
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    for eprop in edges.property_names:
        df_props = _get_props(edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by, eprop)
        _check_props(res, df_props, group_by, eprop)

    ## (c) W/ group-by (src only)
    group_by = ("layer", None)
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    for eprop in edges.property_names:
        df_props = _get_props(edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by, eprop)
        _check_props(res, df_props, group_by, eprop)

    ## (d) W/ group-by (tgt only)
    group_by = (None, "layer")
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    for eprop in edges.property_names:
        df_props = _get_props(edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by, eprop)
        _check_props(res, df_props, group_by, eprop)

    ## (e) W/ group-by (different src/tgt)
    group_by = ("layer", "mtype")
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    for eprop in edges.property_names:
        df_props = _get_props(edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by, eprop)
        _check_props(res, df_props, group_by, eprop)

    # Case 3: Partial circuit (layer by layer) w/ group-by
    group_by = "mtype"
    for _src_lay in nodes[0].property_values("layer"):
        for _tgt_lay in nodes[1].property_values("layer"):
            sel_src = {"layer": _src_lay}
            sel_tgt = {"layer": _tgt_lay}
            res = test_module.compute(
                circuit,
                sel_src=sel_src,
                sel_dest=sel_tgt,
                edges_popul_name=popul_name,
                group_by=group_by,
                skip_empty_groups=True,
            )
            for eprop in edges.property_names:
                df_props = _get_props(
                    edges_table,
                    nodes[0].ids(sel_src),
                    nodes[1].ids(sel_tgt),
                    nodes,
                    group_by,
                    eprop,
                )
                _check_props(res, df_props, group_by, eprop)

    # Case 4: Empty source/target node selection (with and w/o skip)
    group_by = "layer"
    for skip in [True, False]:
        ## (a) src sel empty
        sel_src = {"layer": "UNKNOWN"}
        sel_tgt = None
        res = test_module.compute(
            circuit,
            sel_src=sel_src,
            sel_dest=sel_tgt,
            edges_popul_name=popul_name,
            group_by=group_by,
            skip_empty_groups=skip,
        )
        _check_empty(
            res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, edges, group_by, skip
        )

        ## (b) tgt sel empty
        sel_src = None
        sel_tgt = {"layer": "UNKNOWN"}
        res = test_module.compute(
            circuit,
            sel_src=sel_src,
            sel_dest=sel_tgt,
            edges_popul_name=popul_name,
            group_by=group_by,
            skip_empty_groups=skip,
        )
        _check_empty(
            res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, edges, group_by, skip
        )

        ## (c) src & tgt sel empty
        sel_src = {"layer": "UNKNOWN"}
        sel_tgt = {"layer": "UNKNOWN"}
        res = test_module.compute(
            circuit,
            sel_src=sel_src,
            sel_dest=sel_tgt,
            edges_popul_name=popul_name,
            group_by=group_by,
            skip_empty_groups=skip,
        )
        _check_empty(
            res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, edges, group_by, skip
        )
