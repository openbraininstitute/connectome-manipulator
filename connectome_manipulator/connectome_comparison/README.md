# Structural comparisons overview

Comparing two connectomes in terms of connectivity or synaptic properties.

| File | Description | Example |
| :-- | :-- | :-- |
| __[/adjacency.py](adjacency.py)__ | __Adjacency matrix__ <br> Connectivity between pairs of pre-/postsynaptic neurons <br> Optionally: <br> Numbers of synapses per connection | ![Adjacency matrix](../../doc/source/images/struct_comp_adjacency.png "Adjacency matrix") |
| __[/connectivity.py](connectivity.py)__ | __Connectivity matrix__ <br> Connection probability and #synapses/connection between selected groups of neurons (layers, m-types, ...) | ![Connectivity matrix](../../doc/source/images/struct_comp_conn_prob.png "Connectivity matrix") |
| __[/properties.py](properties.py)__ | __Property matrix__ <br> Mean/std/... values of synapse properties between selected groups of neurons (layers, m-types, ...) | ![Property matrix](../../doc/source/images/struct_comp_prop_delay.png "Property matrix") |


## Grouping behavior for connectivity/property matrices:

- No pre-/postsynaptic grouping (`group_by=None`), i.e., returning a single value

  <img width="128" alt="No grouping" src=../../doc/source/images/conn_prob__group_by_none.png>

- Separate pre-/postsynaptic groupings by providing a tuple, e.g., `group_by=("mtype", "etype")`

  <img width="128" alt="Pre-/postsynaptic grouping" src=../../doc/source/images/conn_prob__group_by_src_tgt.png>

- Omitting grouping, if not existing in either pre- or postsynaptic node population (e.g., in case of virtual populations)
  - e.g., `group_by="layer"` but "layer" only existing in presynaptic population, i.e., turned into `("layer", None)`
 
    <img width="128" alt="Source grouping" src=../../doc/source/images/conn_prob__group_by_src.png>

  - e.g., `group_by="layer"` but "layer" only existing in postsynaptic population, i.e., turned into `(None, "layer")`
 
    <img width="128" alt="Target grouping" src=../../doc/source/images/conn_prob__group_by_tgt.png>
 
  - CAUTION:
    - `group_by="unknown-property"` which is not existing in either population will still raise an error!
    - `group_by=("layer", "layer")` but "layer" only existing in one of the populations will also raise an error!


Copyright (c) 2024 Blue Brain Project/EPFL<br>
Copyright (c) 2025 Open Brain Institute