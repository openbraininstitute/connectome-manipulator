Changelog
=========

Version 1.0.4
-------------

- Improvements for group_by selection:

  - Fixed the special cases when group_by=None is selected for extracting connectivity or synaptic properties
  - Added option to specify group_by separately for source/target node populations as a tuple
  - Improved logic when the group_by property does not exist in either the source or target node population
  - Moved common functions into access_functions.py and utils.py for better reusability
  - Added unit tests for connectivity and properties extraction
  - Small fixes to take changes in new scipy and pyarrow versions into account


Version 1.0.3
-------------

- Added new feature (max. distance for connectivity extraction)
- Small fixes


Version 1.0.2
-------------

- Added new features and examples (cross-validation, synapse parameter correlation)
- Changed afferent section types in accordance with MorphIO (1: soma, 2: axon, 3: basal dendrite, 4: apical dendrite)
- Use of MorphIO collections
- Improved readme and documentation


Version 1.0.1
-------------

- Added Python 3.11 and 3.12 support
- Added documentation building


Version 1.0.0
-------------

- First major release


Version 0.0.11.dev1
-------------------

- New synapse position re-use mode "reuse_strict" with re-use restricted to source selection
- Minor fixes for empty data splits, node selection, and data logs
- Additional examples


Version 0.0.11.dev0
-------------------

- Added offset operation to synapse properties alteration
- Minor fixes


Version 0.0.10
--------------

- New release


Version 0.0.10.dev4
-------------------

- Added license & copyright for open-sourcing
- Added example notebook
- Added version info to log files
