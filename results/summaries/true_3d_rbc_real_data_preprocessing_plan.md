# True 3D RBC Real-Data Preprocessing Plan

This plan prepares real experimental MFL observations for the 20.96 liftoff-conditioned inference runner. It does not train a model, run COMSOL, write NPZ data, or change CURRENT_BASELINE.md.

Main chain: raw Bx/By/Bz -> alignment -> Tesla units -> matched no-defect reference -> delta_b -> 201-sample x grid -> three scan lines -> sensor_z_m metadata -> 20.96 inference runner.

Gain/amplitude calibration remains diagnostic. It may be recorded and compared, but it does not replace the baseline or A2 companion routing contract.

## Steps
1. `read_raw_bxyz`: raw Bx/By/Bz scans or prepared delta_b -> array with axis metadata. Blocker if missing: Bx/By/Bz tri-axis data.
2. `spatial_align_three_axes`: Bx/By/Bz channels -> aligned tri-axis field. Blocker if missing: sensor_alignment_status.
3. `convert_units_to_tesla`: field array and unit metadata -> Tesla-valued field array. Blocker if missing: known unit.
4. `match_no_defect_reference`: defect scan and no-defect reference -> matched reference pair. Blocker if missing: no_defect_reference_id.
5. `compute_delta_b`: b_defect and b_no_defect -> delta_b=b_defect-b_no_defect. Blocker if missing: trusted delta_b or raw pair.
6. `resample_sensor_x`: sensor_x coordinate and signal -> 201 x-samples. Blocker if missing: resampling map to 201.
7. `map_scan_line_y`: scan_line_y metadata -> three scan lines matching [-0.001,0,0.001] convention. Blocker if missing: three-line y mapping.
8. `write_sensor_z_metadata`: measured liftoff -> sensor_z_m per sample. Blocker if missing: sensor_z_m.
9. `record_gain_calibration`: gain/amplitude calibration info -> diagnostic calibration flag. Blocker if missing: not a blocker, but warning if unknown.
10. `call_20_96_inference_runner`: delta_b + sensor_z_m + metadata -> RBC params, profile/depth, projected mask. Blocker if missing: validated intake manifest.
