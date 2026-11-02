LOWERING FACTOR CALIBRATION AND VALIDATION TOOL v1.0
====================================================

PURPOSE
-------
This standalone Tkinter tool calibrates a licence-area dredging production rate
from an existing bathymetric difference raster and EMS-recorded dredging time.
It creates cell-level QA outputs, licence summaries, residuals, and a JSON rate
profile that can later be checked by the EMS lowering tool.

IMPORTANT
---------
Positive standardised change means seabed lowering. Measured bathymetric change
may also include sediment movement, accretion, survey offsets and artefacts.
Always review the spatial residuals and excluded/review cells before operational use.

INPUTS
------
1. An existing bathymetric difference raster. The Validator does not create
   interpolated surfaces from XYZ or paired surface rasters in its GUI.
2. Calibration grid with a unique GridID field. For a GeoPackage with multiple
   layers, choose its path, enter the matching grid ID field, then click
   "Load layers" and select the layer containing that field.
3. Licence polygons with a LicenceNo field. For a multi-layer GeoPackage, click
   its "Load layers" button and select the layer containing the licence field.
   To update the Lowering Model, select its `areas` layer here and use its
   `area_id` as the licence field.
4. EMS vector points, or a CSV/TXT/DAT file or root folder. In the GUI use
   "Select file..." for one EMS file or "Select folder..." for an archive root.
5. Survey dates and QA thresholds.

The preferred difference raster is an ESRI Binary Float Grid (`.flt`) with
matching `.hdr` and `.prj` files. A single-band float32/float64 GeoTIFF is also
accepted. The raster must have a known CRS, usable geotransform, and finite
non-NoData values in metres. Integer grids, multiband RGB/RGBA images, palette
images, and rasters with missing CRS are rejected.

The raster is inspected with rasterio/GDAL immediately after selection and
again before a validation run. This preflight occurs before opening the EMS
database, scanning EMS files, or querying records. It reports raster metadata,
valid/NoData/NaN/infinite pixel counts, value ranges and signs; the median is
computed from a bounded systematic sample on large rasters. Zero is treated
according to the raster's actual NoData/mask metadata and remains a valid
no-change value otherwise. The project units setting must be `metres`; units
declared in raster metadata are checked against it.

Create the survey surfaces and their difference in MapInfo Pro (or another
GIS) before running the Validator.

EMS CSV/TXT/DAT defaults:
  GPSLON, GPSLAT, DATE, TIME, DREDGE
The Validator also accepts LONGITUDE/LATITUDE and a combined DATETIME field.
When DATE and TIME are separate, it combines them. If no vessel field is
present, the source filename is used as the vessel identifier. Field names are
configurable. Delimited EMS data are assumed to be WGS84.
Vector EMS data retain their declared CRS.

ANALYSIS EXTENT
---------------
The default "FULL_OVERLAP" extent starts with the complete calibration grid and
licence polygons, then limits analysis to grid cells within the difference
raster bounds that contain at least one valid raster pixel. Partially covered cells remain available for the
configured raster-coverage QA threshold. Cells without any valid pixels are
left out of the cell-level results and rate calculation.
The raster bounds are also applied while reading the grid and licence layers,
and EMS rows are streamed from the shared index rather than copied into memory
as a full-archive table. EMS scanning progress is reported during large runs.
The active SQLite database path and initialization profile are logged before
EMS source discovery begins.

EMS INDEX
---------
The Validator and EMS Database Manager use a rebuildable SQLite index whose
records are keyed by EMS source root and import profile. The active SQLite file
location is stored in the shared per-user configuration at
`%APPDATA%\EMS_Toolkit\ems_toolkit_config.json`
and can be managed by running `ems_database_manager.py` from this folder. The
manager can be started with `python ems_database_manager.py`. The database
file can live on an external or modelling drive; source-root identity
remains stored in its records and is independent of the database location.
When no database location has been configured, an existing legacy cache may
still be reused. A missing configured database never causes a replacement
cache to be silently created: the Validator offers to locate the existing file
or open the EMS Database Manager.

The EMS Database Manager can create a new database only after an explicit user
action, select an existing compatible EMS database, display its indexed source
roots/profiles and open its folder. It verifies the expected tables, columns,
indexes, R-tree, and metadata without a full integrity scan before activation.
Copying uses SQLite's backup API to include WAL-managed data consistently,
verifies the copy before activation, and retains the original. Close the
Validator and other tools using the database before copying.

CRS values for delimited Validator imports are taken from their known
EPSG:4326 profile configuration. Source-file CRS values are stored in the
`source_files.crs` metadata column for imports that determine CRS from their
records. Existing databases are upgraded with a nullable metadata-only column
when opened; no EMS record scan or rewrite is performed. Existing ready rows
for a known profile can be populated from that profile configuration during
normal file synchronization.

First use imports the EMS records. Later runs reuse unchanged files and refresh
files whose size or modification time changed.
Delimited sources are parsed in chunks and inserted through a database-ready
bulk path, avoiding an EMSRecord object for each imported row; the existing
EMSRecord API remains available to other callers. The R-tree is built with a
file-scoped SQL insert. Per-file timing reports separate input reading, vector
conversion, row/batch creation, SQLite insertion, and R-tree construction.
Imports use a 64 MiB SQLite page cache on their connection; index and query
behavior is unchanged. Reports also separate file cleanup, source metadata,
connection setup, transaction commit, and close time to identify SQLite
overhead outside the main inserts. Validator startup logs separately time
database path setup, WAL and initialization PRAGMAs, schema and per-index
existence/creation checks, metadata reads, commit, and connection close.
Streaming index queries can apply date, dredge, coordinate, and spatial
bounding-box filters. The Validator applies its bounding-box check after
per-vessel time sampling so existing sample results remain unchanged. Unchanged
files previously marked failed are retried on later runs. The database can be
rebuilt from the raw files with EMSDatabase.rebuild, or by removing the cache
file and running either tool.

An experimental R-tree-first benchmark is opt-in with
`EMS_SPATIAL_FIRST_EXPERIMENT=1`. It runs both the current query/sampling
pathway and an R-tree-prefiltered pathway, reports query plans, first-row and
streaming timings, candidate/sample counts, and retained-record differences.
It is deliberately comparison-only: the Validator continues using the
current pathway's records and results. Spatial prefiltering can change which
record wins a time bin, so do not use the experimental output as a validation
result. In FULL_OVERLAP mode, its envelope is based on the raster-bounded
master-grid features already loaded by the Validator, preserving whole grid
cell extents at raster edges.

For a one-off count of rows matching the Validator's current date, dredge, and
coordinate filters, set `EMS_QUERY_COUNT_DIAGNOSTIC=1` before launching the
Validator. This performs a separate unordered SQLite COUNT and reports its
elapsed time; it is disabled by default because broad counts can still take
time on a large archive. The ordered record iterator reports time to first row,
100,000 rows, 1,000,000 rows, and completion. Its optional cancellation event
uses SQLite's progress handler and does not change query behavior otherwise.

RASTER SIGN
-----------
The raster value convention is applied first: ELEVATION values are used as
they are, and POSITIVE_DEPTH values are negated so they behave like elevations.
The lowering sign then says which sign means seabed lowering AFTER that step.

  Raster                                   Convention       Lowering sign
  Elevation, earlier minus later           ELEVATION        positive_is_lowering
  Elevation, later minus earlier           ELEVATION        negative_is_lowering
  Positive depth, later minus earlier      POSITIVE_DEPTH   negative_is_lowering
  Positive depth, earlier minus later      POSITIVE_DEPTH   positive_is_lowering

Check the "Bathymetric difference convention" line in the raster summary: it
states which raw raster sign will be treated as lowering. The Validator does
not infer a sign, take absolute values, or silently invert data. The output
ObsLower_m is always positive for lowering.

CALCULATIONS
------------
Observed volume = observed lowering x valid raster-covered area
Observed m3/hour = observed volume / EMS hours
Observed t/hour = observed m3/hour x bulk density
Observed lowering factor = observed lowering / EMS hours

Licence rate is calculated from total accepted volume / total accepted EMS hours,
then converted to tonnes/hour using the entered bulk density.

BACKGROUND (NATURAL) LOWERING
Cells with no EMS hours still change: natural seabed movement, or a small
vertical offset between the two surveys. That change also happens between the
narrow (about 2 m) dredge furrows inside dredged cells. With
background_correction = UNDREDGED_CELLS (default) the Validator takes the
median lowering of undredged cells (no EMS hours, valid bathymetry, coverage at
or above the minimum) and subtracts it from every cell before volumes and
rates. Each licence uses its own undredged cells when it has at least
background_min_cells (default 5); otherwise all licences are pooled; with too
few cells overall no correction is applied and the log warns. The log also
warns when the undredged-cell spread (IQR) is larger than the median. Set
NONE to treat all lowering as dredging.
Columns: RawObsLower_m (before correction), Background_m, ObsLower_m (after).

RATE METHOD
POOLED (default): the licence rate is total corrected volume / total EMS hours
over every cell with at least the minimum EMS hours and raster coverage.
Detection-threshold, non-positive-rate and maximum-rate checks are reported in
QA_Flags per cell but do not remove cells, because averaged over a 250 m cell
random survey error is small and dropping lightly dredged cells (with their
hours) biases the rate upward. CELL_FILTERS: the earlier behaviour, where those
checks also reject cells.

Summary checks: UncorrectedRate_tph (same cells, no background correction),
RegressionRate_tph (slope of raw cell volume against EMS hours; it ignores any
constant background), AllEMSCellsRate_tph (every cell with EMS time and valid
bathymetry), AllEMSCellsHours and ExcludedEMSHoursPct. The log warns when the
filters remove more than 25% of a licence's EMS hours.

Per-cell rates (ObsRate_tph) are a QA map only: with a 2 m footprint and the
EMS position some distance from the draghead, cells with few hours are noisy.
Treat 50 m results as exploratory for the same reason.

EMS hours = sampled records x EMS sample interval. The sample interval must
match how often EMS logs records. The log reports the median gap between
consecutive active records and warns when it is more than 1.5x the configured
interval, because that would overstate every rate.

QA CLASSES
----------
A_ACCEPTED: passes all configured QA thresholds.
B_REVIEW: has some matched data but fails one or more thresholds.
C_EXCLUDED: has no sufficient matched EMS or bathymetric information.

STATUS
------
GREEN: validated and configured rates differ by no more than GREEN tolerance.
AMBER: difference exceeds GREEN but not AMBER tolerance.
RED: difference exceeds AMBER tolerance.
INSUFFICIENT_DATA: no accepted cell/time basis for a rate.

OUTPUTS
-------
lowering_factor_validation_grid.gpkg
  calibration_grid_all
  calibration_grid_accepted
  calibration_grid_review
  calibration_grid_excluded
  licence_summary

lowering_factor_validation_summary.csv
lowering_rate_profile.json
lowering_factor_validation_report.txt
processing_log.txt
run_log.txt  (whole run: settings, pre-run checks, EMS database steps and
              every message, written even when a run is cancelled or fails)

RUNNING AND RE-RUNNING
----------------------
Reuse EMS data (ticked by default): the sampled EMS records are saved in
<output folder>\ems_cache after a run. A later run with the same EMS folder,
field names, dredge filter, survey dates and sample interval, over the same or a
smaller grid extent, loads them instead of querying the EMS database. Untick it
after adding or changing EMS files. Command line: --no-ems-cache.

Licences to run: leave blank for all, or type IDs separated by commas, or use
"Choose..." to pick from the licence layer. Only grid cells inside the chosen
licences are used. Command line: --licences "137, 522".

Pre-run checks run before the slow EMS stage and report: overlapping grid
cells that belong to the same licence (counted twice; overlaps involving
cells outside every licence, or across a boundary between neighbouring
licences, are only noted in the log), licences with no grid cells, blank or reversed survey
dates, calendar months in the survey period with no EMS records at all, and a
raster whose sign settings make few pixels read as lowering. The app asks
whether to continue; the command line logs them and continues.

While running, the window shows the current stage and elapsed time. Cancel
stops at the next safe point (an EMS file being imported is rolled back).
When the run finishes, a results window lists each licence with the validated,
uncorrected and regression rates, background, cells and hours used, with
buttons to open the output folder, summary CSV and processing log. "Show last
results" reopens it.

INSTALLATION
------------
Store all files locally. From a command prompt in this folder:

  python -m pip install -r requirements.txt
  python lowering_factor_validator.py

A managed GIS Python environment may already include some dependencies. Tkinter is
included with most standard Windows Python installations.

PRACTICAL FIRST RUN
-------------------
Use a known 250 m calibration grid and an already quality-controlled difference
raster. Ensure the survey start/end dates exactly describe the comparison
period, and confirm that the raster CRS and vertical units are appropriate.

MULTI-RESOLUTION VALIDATION
---------------------------
The 250 m calibration grid remains the authoritative regulatory master. The
resolution selector defaults to "250 m Regulatory"; the former "50 m Internal
Detail" setting is accepted as a legacy alias for the explicitly experimental
"50 m Experimental" option. "Both" runs 250 m Regulatory and 50 m Experimental.
Refinement retains the master CRS and footprint, records each parent ID and
child row/column, and requires a projected metre CRS plus a verifiable 250 m
affine lattice. Rotated square grids are supported. For 50 m runs, alignment
is established from the original master-grid layer before raster-bounds
filtering. If source features are clipped, the tool accepts them only when
their boundaries uniquely establish a 250 m lattice and each feature is
contained in exactly one cell; generated children use the complete aligned
parent cell. Ambiguous or misaligned grids are rejected rather than
independently re-aligned. A 250 m-only run does not infer the child-grid
lattice because it does not generate children; the selected layer should
still be the intended 250 m regulatory master.

Each selected resolution is validated independently against the same survey
difference surface and one EMS load/sample. The 50 m grid is generated as 5 x 5
children within each authoritative 250 m parent. Child IDs are stable from the
parent ID and child row/column; outputs retain Parent250_ID, Child50_ID and the
legacy ParentGridID/row-column fields. The 50 m path uses the same raster
statistics, EMS sample interval, calculation formulas and rejection thresholds
as the existing 250 m method. It is an experiment only and never supplies
source-model lowering-factor write-back values.

The 50 m experiment reports EMS-duration and pre-maximum positive-rate
distributions, threshold/rejection counts, stage timings, and experimental
parent-250 m aggregation. Parent rates are derived from summed valid child
volume divided by the summed EMS duration of valid raster-covered,
EMS-bearing children; child rates are not simply averaged. Outputs include
validation_50m_diagnostics.csv,
validation_50m_vs_250m_summary.csv,
validation_50m_parent_250m_aggregation.csv, and a resolution-specific
GeoPackage. Running Both retains separate 250 m Regulatory and 50 m
Experimental output names; a 250 m-only run preserves the established output
names.

The 50 m geometry is generated from the loaded 250 m master extent, which is
restricted to raster coverage in FULL_OVERLAP mode, then follows the same
licence assignment and per-resolution raster-coverage checks as 250 m. Both
grids use the same once-loaded sampled EMS population. For 50 m only, points
on shared child boundaries are assigned once using ascending child row/column
and then Child50_ID as deterministic tie-breaks; the 250 m Regulatory
assignment rule remains unchanged.

The Lowerer uses its own resolution choices and keeps 250 m as its default.
Its regulatory master-size setting must be 250 m and is checked against the
input geometry. Selecting 250 m only does not build child features; 50 m only
allocates the sampled EMS records directly to the generated 50 m polygons;
Both allocates the same once-loaded sample independently to each grid. Existing
cumulative-grid updates remain a 250 m operation.

When Both is selected in the Lowerer, resolution_volume_conservation_qc.csv
compares parent and child volumes from unrounded lowering depth multiplied by
the nominal cell area established by the verified 250 m/50 m lattice. The
documented absolute tolerance is 0.000001 m3; output values are not adjusted.
Clipped child geometries inherit the master's valid footprint, while lowering
continues to use the nominal cell area as in the existing cell-based method.
Review failed comparisons for boundary assignments, masked cells, duplicate or
omitted EMS samples, or inconsistent filters/CRS.

LIMITATIONS IN VERSION 1.0
--------------------------
- Raster valid area is estimated from valid pixel count and pixel area.
- EMS interval time is represented by retained sample count x sampling interval.
- Each grid cell is assigned to the licence containing its representative point.
- Grid cells that are not complete 250 x 250 m squares (within 1 m on each
  side and 200 m2 in area), such as cells clipped to a survey or licence
  boundary, are removed before both the 250 m and 50 m runs. The log lists the
  removed GridIDs. If no complete cells remain, the run stops.
- The JSON profile is not cryptographically signed.

LOWERING MODEL FACTOR UPDATE
----------------------------
Updating factors later: tick "Replace existing factors for the licences in
this run" (command line --replace-model-factors). Only the licences in that run
are changed; the app lists each old -> new factor and asks before replacing
(answering No still fills blank factors). Every fill and replacement is
recorded in the model's lowering_factor_history table (date, area, old and new
factor, validated m3/h, background, cells, hours, survey dates, rate method,
output folder), so earlier factors can be restored. All changes in a run are
written in one transaction.

Write-back is OFF by default. Switch on "Fill blank lowering_factor values"
in the GUI, set "write_model_factors": true in a settings file, or pass
--write-model-factors on the command line. Review the results first: a filled
factor is never overwritten by later runs. The log lists each candidate rate
with its QA status. (QA status compares against the configured rate, so a RED
rate is not necessarily wrong data.)

Where it writes: set "Model to update" to the Lowering Model GeoPackage
(model layer `areas`, model area ID field `area_id`) to keep using the licence
polygons (e.g. All_ADZ) for the calculation while filling the model's factors;
each licence number is matched to the model's area_id. Leave "Model to update"
blank to write to the licence polygon layer instead. Command line:
--model-path, --model-layer, --model-id-field.

When write-back is on and the target layer is a GeoPackage containing a
`lowering_factor` field, the Validator writes its 250 m regulatory result
to that layer. It matches the validation licence ID to the configured licence
field (for the Lowering Model, `areas.area_id`) and stores
`ValidatedRate_m3ph / 62,500` as metres/hour, matching the Lowerer's factor for
a 250 m cell. Existing nonblank factor values are preserved; only blank or NULL
values are filled. In "Both" mode, the 250 m result is used. A 50 m-only run
does not update the model factor. If the selected layer is not a GeoPackage or
has no `lowering_factor` field, write-back is skipped and logged.
