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

import connectome_manipulator.connectome_comparison.connectivity as test_module


def _get_conn(edges_table, src_ids, tgt_ids, nodes, group_by):
    """Extract connectivity matrices from edges table"""
    conns, cnts = np.unique(
        edges_table[["@source_node", "@target_node"]], axis=0, return_counts=True
    )
    adj_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=bool)
    cnt_mat = np.zeros((len(src_ids), len(tgt_ids)), dtype=float) * np.nan
    for (_s, _t), _c in zip(conns, cnts):
        adj_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = True
        cnt_mat[np.where(src_ids == _s)[0], np.where(tgt_ids == _t)[0]] = _c

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

    df_prob = pd.DataFrame(adj_mat, columns=tgt_grp)
    df_prob["src"] = src_grp
    df_prob = df_prob.melt("src", var_name="tgt", value_name="data")
    df_prob = df_prob.groupby(["src", "tgt"]).mean() * 100  # in %

    df_nsyn = pd.DataFrame(cnt_mat, columns=tgt_grp)
    df_nsyn["src"] = src_grp
    df_nsyn = df_nsyn.melt("src", var_name="tgt", value_name="data")

    df_nsyn_mean = df_nsyn.groupby(["src", "tgt"]).mean().fillna(0.0)
    df_nsyn_std = df_nsyn.groupby(["src", "tgt"]).std(ddof=0).fillna(0.0)
    df_nsyn_sem = df_nsyn.groupby(["src", "tgt"]).sem(ddof=0).fillna(0.0)
    df_nsyn_min = df_nsyn.groupby(["src", "tgt"]).min().fillna(0.0)
    df_nsyn_max = df_nsyn.groupby(["src", "tgt"]).max().fillna(0.0)

    return (
        df_prob.unstack(),
        df_nsyn_mean.unstack(),
        df_nsyn_std.unstack(),
        df_nsyn_sem.unstack(),
        df_nsyn_min.unstack(),
        df_nsyn_max.unstack(),
    )


def _check_conn(
    res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
):
    for _key in [
        "conn_prob",
        "nsyn_conn",
        "nsyn_conn_std",
        "nsyn_conn_sem",
        "nsyn_conn_min",
        "nsyn_conn_max",
        "common",
    ]:
        assert _key in res, f'ERROR: Results key "{_key}" missing!'
    for _key in ["src_group_values", "tgt_group_values"]:
        assert _key in res["common"], f'ERROR: Results key "{_key}" in "common" missing!'
    for _key in [
        "conn_prob",
        "nsyn_conn",
        "nsyn_conn_std",
        "nsyn_conn_sem",
        "nsyn_conn_min",
        "nsyn_conn_max",
    ]:
        assert "data" in res[_key], f'ERROR: Results key "data" in "{_key}" missing!'

    if not isinstance(group_by, tuple):
        group_by = (group_by, group_by)
    if group_by[0] is None:
        assert_array_equal(
            res["common"]["src_group_values"], [None], "ERROR: Source group mismatch!"
        )
    else:
        assert_array_equal(
            res["common"]["src_group_values"],
            df_prob.index.to_numpy(),
            "ERROR: Source group mismatch!",
        )
    if group_by[1] is None:
        assert_array_equal(
            res["common"]["tgt_group_values"], [None], "ERROR: Target group mismatch!"
        )
    else:
        assert_array_equal(
            res["common"]["tgt_group_values"],
            df_prob.columns.get_level_values(1).to_numpy(),
            "ERROR: Target group mismatch!",
        )
    assert_allclose(
        res["conn_prob"]["data"], df_prob.to_numpy(), err_msg="ERROR: Conn. prob. mismatch!"
    )
    assert_allclose(
        res["nsyn_conn"]["data"],
        df_nsyn_mean.to_numpy(),
        err_msg="ERROR: Nsyn/conn (mean) mismatch!",
    )
    assert_allclose(
        res["nsyn_conn_std"]["data"],
        df_nsyn_std.to_numpy(),
        err_msg="ERROR: Nsyn/conn (std) mismatch!",
    )
    assert_allclose(
        res["nsyn_conn_sem"]["data"],
        df_nsyn_sem.to_numpy(),
        err_msg="ERROR: Nsyn/conn (sem) mismatch!",
    )
    assert_allclose(
        res["nsyn_conn_min"]["data"],
        df_nsyn_min.to_numpy(),
        err_msg="ERROR: Nsyn/conn (min) mismatch!",
    )
    assert_allclose(
        res["nsyn_conn_max"]["data"],
        df_nsyn_max.to_numpy(),
        err_msg="ERROR: Nsyn/conn (max) mismatch!",
    )


def _check_empty(res, src_ids, tgt_ids, nodes, group_by, skip):
    nsrc = len(nodes[0].property_values(group_by))
    ntgt = len(nodes[1].property_values(group_by))
    res_keys = [
        "nsyn_conn",
        "nsyn_conn_std",
        "nsyn_conn_sem",
        "nsyn_conn_min",
        "nsyn_conn_max",
        "conn_prob",
    ]
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


def test_connectivity():
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
    df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
        edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by
    )
    _check_conn(
        res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
    )

    ## (b) W/ group-by
    group_by = "layer"
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
        edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by
    )
    _check_conn(
        res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
    )

    ## (c) W/ group-by (src only)
    group_by = ("layer", None)
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
        edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by
    )
    _check_conn(
        res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
    )

    ## (d) W/ group-by (tgt only)
    group_by = (None, "layer")
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
        edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by
    )
    _check_conn(
        res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
    )

    ## (e) W/ group-by (different src/tgt)
    group_by = ("layer", "mtype")
    res = test_module.compute(
        circuit, sel_src=None, sel_dest=None, edges_popul_name=popul_name, group_by=group_by
    )
    df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
        edges_table, nodes[0].ids(), nodes[1].ids(), nodes, group_by
    )
    _check_conn(
        res, df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max, group_by
    )

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
            df_prob, df_nsyn_mean, df_nsyn_std, df_nsyn_sem, df_nsyn_min, df_nsyn_max = _get_conn(
                edges_table, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, group_by
            )
            _check_conn(
                res,
                df_prob,
                df_nsyn_mean,
                df_nsyn_std,
                df_nsyn_sem,
                df_nsyn_min,
                df_nsyn_max,
                group_by,
            )

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
        _check_empty(res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, group_by, skip)

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
        _check_empty(res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, group_by, skip)

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
        _check_empty(res, nodes[0].ids(sel_src), nodes[1].ids(sel_tgt), nodes, group_by, skip)
