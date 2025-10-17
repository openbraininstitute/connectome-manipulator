# This file is part of connectome-manipulator.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Blue Brain Project/EPFL
# Copyright (c) 2025 Open Brain Institute

"""Module for comparing connectomes based on synapse properties:

Structural comparison of two connectomes in terms of synapse properties per pathway,
as specified by the config. For each connectome, the underlying properties maps are
computed by the :func:`compute` function and will be saved to a data file first.
The individual synapse properties maps, together with a difference map between the
two connectomes, are then plotted by means of the :func:`plot` function.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
from connectome_manipulator.access_functions import get_edges_population, get_node_ids, get_grouping
from connectome_manipulator.utils import check_grouping


def compute(
    circuit,
    fct="np.mean",
    group_by=None,
    sel_src=None,
    sel_dest=None,
    per_conn=False,
    skip_empty_groups=False,
    edges_popul_name=None,
    **_,
):
    """Computes a matrix of synapse property values between groups of neurons of a given circuit's connectome.

    Args:
        circuit (bluepysnap.Circuit): Input circuit
        fct (str): Function to apply, e.g., "np.mean", "np.std"
        group_by (str/tuple): Neuron property name based on which to group connections, e.g., "synapse_class", "layer", or "mtype"; can be a tuple with two property names for source/target neurons; if omitted, the overall average is computed
        sel_src (str/list-like/dict): Source (pre-synaptic) neuron selection
        sel_dest (str/list-like/dict): Target (post-synaptic) neuron selection
        per_conn (bool): If selected, ``fct`` is applied to the average property value per connection (i.e., average value of all synapses belonging to a connection); otherwise, ``fct`` is applied to the synapses of all connections altogether
        skip_empty_groups (bool): If selected, only group property values that exist within the given source/target selection are kept; otherwise, all group property values, even if not present in the given source/target selection, will be included
        edges_popul_name (str): Name of SONATA egdes population to extract data from

    Returns:
        dict: Dictionary containing the computed data elements; see Notes

    Note:
        The returned dictionary contains the data elements that can be selected for plotting through the structural comparison configuration file, together with a common dictionary containing additional information. Each data element is a dictionary with "data" (numpy.ndarray of size <source-group-size x target-group-size>), "name" (str), and "unit" (str) items. Names of these data elements correspond to the synapse properties that are present in the given SONATA edges population. Usual properties may include for example:

        * "conductance": Peak conductance
        * "decay_time": Decay time constant
        * "depression_time": Time constant for recovery from depression
        * "facilitation_time": Time constant for recovery from facilitation
        * "u_syn": Utilization of synaptic efficacy
        * ...
    """
    # Select edge population
    edges = get_edges_population(circuit, edges_popul_name)

    # Select corresponding source/target nodes populations
    src_nodes = edges.source
    tgt_nodes = edges.target

    # Get grouping selection
    src_group_by, tgt_group_by = check_grouping(group_by, src_nodes, tgt_nodes)
    src_group_sel, src_group_values = get_grouping(
        src_nodes, sel_src, src_group_by, skip_empty_groups
    )
    tgt_group_sel, tgt_group_values = get_grouping(
        tgt_nodes, sel_dest, tgt_group_by, skip_empty_groups
    )

    # Preselect neurons
    src_ids_base = get_node_ids(src_nodes, sel_src)
    tgt_ids_base = get_node_ids(tgt_nodes, sel_dest)

    if len(src_ids_base) == 0 or len(tgt_ids_base) == 0:
        print("WARNING: Empty source/target node selection(s)!")

    print(
        f"INFO: Extracting synapse properties (group_by={group_by}, sel_src={sel_src}, sel_dest={sel_dest}, N={len(src_group_values)}x{len(tgt_group_values)} groups, per_conn={per_conn})",
        flush=True,
    )

    edge_props = sorted(edges.property_names)
    print(f"INFO: Available synapse properties: \n{edge_props}", flush=True)

    prop_fct = eval(fct)
    prop_tables = np.full((len(src_group_sel), len(tgt_group_sel), len(edge_props)), np.nan)
    pbar = progressbar.ProgressBar()
    for idx_pre in pbar(range(len(src_group_sel))):
        sel_pre = src_group_sel[idx_pre]
        for idx_post, _ in enumerate(tgt_group_sel):
            sel_post = tgt_group_sel[idx_post]
            pre_ids = get_node_ids(src_nodes, sel_pre)  # Grouping selection
            post_ids = get_node_ids(tgt_nodes, sel_post)  # Grouping selection
            pre_ids = np.intersect1d(pre_ids, src_ids_base)  # Merge with base selection
            post_ids = np.intersect1d(post_ids, tgt_ids_base)  # Merge with base selection
            e_sel = edges.pathway_edges(pre_ids, post_ids, edge_props)
            if e_sel.size > 0:
                if per_conn:  # Apply prop_fct to average value per connection
                    conn, conn_idx = np.unique(
                        e_sel[["@source_node", "@target_node"]], axis=0, return_inverse=True
                    )
                    c_sel = pd.DataFrame(index=range(conn.shape[0]), columns=edge_props)
                    for cidx in range(conn.shape[0]):
                        c_sel.loc[cidx, :] = np.mean(e_sel[conn_idx == cidx], axis=0)
                    prop_tables[idx_pre, idx_post, :] = prop_fct(c_sel.to_numpy(), axis=0)
                else:
                    prop_tables[idx_pre, idx_post, :] = prop_fct(e_sel.to_numpy(), axis=0)

    fname = prop_fct.__name__[0].upper() + prop_fct.__name__[1:]
    cname = " (per conn)" if per_conn else ""
    all_res_dict = {
        edge_props[idx]: {
            "data": prop_tables[:, :, idx],
            "name": f'"{edge_props[idx]}" property',
            "unit": f"{fname} {edge_props[idx]}{cname}",
        }
        for idx in range(len(edge_props))
    }
    all_res_dict["common"] = {
        "src_group_values": src_group_values,
        "tgt_group_values": tgt_group_values,
        "src_group_by": src_group_by,
        "tgt_group_by": tgt_group_by,
    }

    return all_res_dict


def plot(
    res_dict, common_dict, fig_title=None, vmin=None, vmax=None, isdiff=False, group_by=None, **_
):  # pragma:no cover
    """Plots a properties matrix or a difference matrix.

    Args:
        res_dict (dict): Results dictionary, containing selected data for plotting; must contain a "data" item with a properties matrix of type numpy.ndarray of size  <#source-group-values x #target-group-values>, as well as "name" and "unit" items containing strings.
        common_dict (dict): Common dictionary, containing additional information; must contain "src_group_values" and "tgt_group_values" items containing lists of source/target values of the grouped property, matching the size of the properties matrix in ``res_dict``
        fig_title (str): Optional figure title
        vmin (float): Minimum plot range
        vmax (float): Maximum plot range
        isdiff (bool): Flag indicating that ``res_dict`` contains a difference matrix; in this case, a symmetric plot range is required and a divergent colormap will be used
        group_by (str/tuple): Neuron property name based on which to group connections, e.g., "synapse_class", "layer", or "mtype"; can be a tuple with two property names for source/target neurons; if omitted, the overall average is computed
    """
    if isdiff:  # Difference plot
        assert -1 * vmin == vmax, "ERROR: Symmetric plot range required!"
        cmap = "PiYG"  # Symmetric (diverging) colormap
    else:  # Regular plot
        cmap = "hot_r"  # Regular colormap

    plt.imshow(res_dict["data"], interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    if fig_title is None:
        plt.title(res_dict["name"])
    else:
        plt.title(fig_title)

    if "src_group_by" in common_dict and "tgt_group_by" in common_dict:
        src_group_by = common_dict["src_group_by"]
        tgt_group_by = common_dict["tgt_group_by"]
    else:
        # Backward compatibility
        src_group_by, tgt_group_by = check_grouping(group_by)

    src_lbl = tgt_lbl = "(all)"
    if src_group_by:
        src_lbl = str(src_group_by)
    if tgt_group_by:
        tgt_lbl = str(tgt_group_by)
    plt.xlabel(f"Postsynaptic {tgt_lbl}")
    plt.ylabel(f"Presynaptic {src_lbl}")

    n_grp = np.maximum(len(common_dict["src_group_values"]), len(common_dict["tgt_group_values"]))
    font_size = max(13 - n_grp / 6, 1)  # Font scaling
    if len(common_dict["src_group_values"]) > 0:
        plt.yticks(
            range(len(common_dict["src_group_values"])),
            common_dict["src_group_values"],
            rotation=0,
            fontsize=font_size,
        )

    if len(common_dict["tgt_group_values"]) > 0:
        if max(len(str(grp)) for grp in common_dict["tgt_group_values"]) > 1:
            rot_x = 90
        else:
            rot_x = 0
        plt.xticks(
            range(len(common_dict["tgt_group_values"])),
            common_dict["tgt_group_values"],
            rotation=rot_x,
            fontsize=font_size,
        )

    cb = plt.colorbar()
    cb.set_label(res_dict["unit"])
