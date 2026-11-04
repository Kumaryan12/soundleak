# Review the public recordings

Select **Public recording review** in the sidebar. The workbench's **Review public recordings** button opens the same page. Existing detector settings and the 40-recording screening result remain unchanged.

1. Choose a recording or use **Previous / Next**. The page initially selects `N394FX_R20`, one of the two recordings without proposals. **Review queue → No proposals** limits the list to it and `N1387U_R44`. Other queues show unreviewed, reviewed or follow-up recordings.
2. Listen to the original WAV and inspect the spectrogram. White numbered lines identify retained proposal ridges; the selected proposal appears in cyan. Time and frequency sliders change the display, and the time selection sets the playback interval. These controls do not rerun or tune detection. A selected proposal also has a separate playback segment and a detailed ridge plot.
3. Record whether an aircraft is audible, any potential issues and your notes. These are human judgments; a recording without proposals does not automatically receive a miss reason.
4. Label each proposal in the editable table. Save before switching recordings. **Reload saved notes** discards unsaved edits and restores the last saved review.
5. Open **Saved review table and exports** to inspect all 40 statuses and download JSON or CSV. Every unreviewed proposal and both recordings without proposals remain in the export.

## Proposal labels

| Label | Intended judgment |
|---|---|
| Unreviewed | No judgment has been saved. |
| Follows aircraft tone | The reviewer hears an aircraft and sees the ridge follow its tone. This does not confirm an ADS-B identity. |
| False proposal | The pattern appears spurious or associated with a different sound. |
| Fragmented track | The retained ridge covers only part of the audible tonal passage. |
| Duplicate harmonic | Another retained proposal describes the same passage at a related harmonic. Select that other proposal in **Related proposal**; self-references are rejected. |
| Unclear | The recording does not support a confident review decision. |

**Reviewed recordings** counts the explicit recording status. **Reviewed proposals** counts proposals assigned a label other than Unreviewed. They measure review progress, not detection accuracy. A recording may need follow-up while some individual proposals already have labels.

## Saved data and reconstruction

Review data lives under `outputs/reviews/public-aircraft/<screen-report-SHA256>/`. Separate `annotations/<recording-id>.json` files store labels, notes, issue flags, input hashes and save timestamps. Writes replace complete files atomically. A stale browser session cannot overwrite newer notes without reloading. No default human labels or fabricated judgments are inserted.

The original screening report stored proposal summaries, without the ridge arrays needed for plotting. On first access, Soundleak reads the original WAV from its ZIP, validates its full member CRC and frozen WAV SHA-256, and uses the saved detector configuration. It requires the original audio/discovery code hashes. Reconstructed timing, frequencies, grouping, counts and fit diagnostics must match the saved metadata within a small numerical tolerance. A mismatch stops preparation. This is verified reconstruction of summary results; it cannot establish array-level identity with measurements that were never archived.

Verified audio, full-recording spectrograms and representative ridge arrays are cached under `assets/<recording-id>/`, with a measurement-file digest and provenance manifest. Prepared assets can be viewed later without rerunning the detector. The cache retains one representative ridge per grouped proposal, matching the original detector's output; it does not recreate discarded harmonic candidates.

Prepare all review assets ahead of opening the UI:

```sh
.venv/bin/python -m soundleak.cli prepare-public-review
```

This was executed for **all 40 recordings and 54 proposals**. The original screening report was preserved. Human review starts with all items unreviewed. A different screening report gets a different directory, so annotations cannot silently attach to a changed run.

The recordings are from [Praveen Pathak's public dataset](https://zenodo.org/records/23424278), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The page plays the original WAV and displays locally computed spectrograms. [Dataset qualification limits](public-datasets.md) still apply: these reviews do not establish independent receiver-location accuracy.
