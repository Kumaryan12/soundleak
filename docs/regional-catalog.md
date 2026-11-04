# Regional aircraft catalog experiment

The broader catalog substantially reduced missing flight coverage for the existing one-hour development recording. The 32 acoustic proposals, review selection, spatial grid, clock grid, source-tone fitting, sound speed, receiver elevation and scoring thresholds were held fixed. Both original flight segments were retained with exactly the same timestamps and coordinates.

| Measure | Original two-aircraft pool | Regional catalog |
|---|---:|---:|
| Candidate flight segments | 2 | 730 |
| Matched acoustic groups | 3 | 31 |
| Unmatched acoustic groups | 29 | 1 |
| Groups without complete candidate-track coverage | 14 | 0 |
| Distinct matched flight segments at top hypothesis | 2 | 30 |
| Capped objective, Hz | 2.88685 | 1.70175 |
| Best grid position, meters east/north of declared center | 500 / 0 | 500 / 0 |
| Shared clock adjustment, seconds | 0 | +0.5 |
| Outcome | Ambiguous | Ambiguous |

The top spatial point stayed the same, but a separated location still had a similar score under the existing ambiguity rule. A larger candidate pool can improve residuals by chance. These results do not confirm all 31 associations or establish independent location accuracy. The top point is about 186 m from the rounded public camera marker; that distance remains a development diagnostic with unquantified microphone-position uncertainty.

## Acquisition and selection

Source: [ADSB.lol archive for 8 September 2026](https://github.com/adsblol/globe_history_2026/releases/tag/v2026.09.08-planes-readsb-staging-0), distributed under ODbL-1.0. The two split tar assets total 3,874,212,864 bytes. Downloads use bounded resumable HTTP ranges, and each assembled part is checked against the publisher's SHA-256 digest before use.

| Archive part | Published SHA-256 |
|---|---|
| `.tar.aa` | `c2d4a8ed715d4e39cb137991857a46af6b33c96bf425cdbc66cfcc979d16fdf4` |
| `.tar.ab` | `4e73ee4880daf3cc1976c85061213caeeeaf398714438434cea7b2fa6b555f07` |

The extractor scanned all **79,044** native aircraft trace records. Selection required at least one reported airborne position inside a **75 km horizontal radius** around the declared search center, during the recording interval padded by 200 seconds before and 60 seconds after. No aircraft identity, aircraft type, detected event time or frequency was used to select records.

The scan selected **628 aircraft records**, with **zero scan/parsing errors**. Geometric-height and trajectory-quality checks yielded **730 usable segments from 587 aircraft**, containing **197,173 position samples**. The audit lists **1,726 rejected segments/fragments**. A rejected fragment does not necessarily exclude the entire aircraft; an aircraft can have several usable and unusable portions.

This is all matching reported data within the declared selection, not a guarantee that every aircraft in the airspace was observed. Missing geometric altitude, stale positions, gaps, short fragments and inconsistent motion affect usability. Barometric altitude is not silently substituted for geometric altitude. The same constant geoid approximation and importer used by the original case were retained.

Native records, hashes, archive-member offsets, selection parameters and source attribution are stored under `data/real/regional/catalog-75km/`. `manifest.json` marks completion only after the full archive scan. The original two-aircraft case is preserved under `outputs/cases/reference-automatic/`.

## Frozen comparison and search

The rerun loaded the existing saved measurements; it did not rerun detection or edit event windows. The baseline report and measurement archive hashes, an array-level acoustic fingerprint, retained baseline segments and fixed settings are embedded in the new case's `provenance.catalog_comparison`.

The expanded search evaluated **69,207,453 scalar-equivalent models**, batching clock offsets to reduce Python overhead. Scalar/batched parity was checked on the original real case and in regression tests. Every candidate passing the same conservative time-coverage sieve was evaluated; the budget was explicitly increased for the larger catalog. This run took approximately **371 seconds** on the development machine.

The case is saved as `outputs/cases/reference-regional/`. The workbench opens this case by default once its integrity manifest is complete. It includes the comparison table, catalog and rejection audit, eight retained spatial hypotheses, event/flight overlays and exact saved measurements. Selecting the regional catalog supports later comparisons with an explicit computation budget.

```sh
.venv/bin/python -m soundleak.cli fetch-regional --workers 12
.venv/bin/python -m soundleak.cli expand-catalog
```

Downloads and a completed matching catalog are reused. Saved case directories are immutable; use a different `--output` to run another experiment. Roughly 4 GB is needed for the archive, with additional temporary space during range assembly. The acoustic and location inputs remain source-informed development inputs.

## Real background and wrong-data checks

Six **contiguous 90-second** excerpts were selected from the six environmental recordings in [AeroSonicDB v1.1.2](https://zenodo.org/records/10215080), by Blake Downward, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Each excerpt came from the earliest 100-second run labeled no-aircraft, with five-second margins. This rule was fixed before scanning. Clips were not stitched, repeated or selected based on detector output.

| Source WAV | Label column | Selected source interval | Acoustic proposals |
|---|---:|---:|---:|
| `1_AUDIO.wav` | 0 | 330–420 s | 0 |
| `2_AUDIO.wav` | 1 | 5–95 s | 0 |
| `3_AUDIO.wav` | 2 | 120–210 s | 0 |
| `4_AUDIO.wav` | 3 | 130–220 s | 0 |
| `5_AUDIO.wav` | 4 | 5–95 s | 0 |
| `6_AUDIO.wav` | 5 | 5–95 s | 0 |

The detector proposed no events across these **540 seconds** of annotated background. Excerpts were resampled locally using the regular decoder. Selection indices, input hashes, annotations and source rights are recorded in `data/real/controls/manifest.json`. Partial ZIP-member extraction uses HTTPS and retains excerpt hashes; it does not claim full ZIP/member CRC verification.

A second check shifted every regional candidate timestamp by one day while retaining the frozen acoustic groups and search settings. The result was **insufficient evidence**, with **zero retained location hypotheses**. This checks missing-time-coverage rejection; it does not test rejection of an incorrect flight with a similar in-window Doppler curve.

```sh
.venv/bin/python -m soundleak.cli background-controls --fetch
.venv/bin/python -m soundleak.cli rejection-checks
```

Reports are saved under `outputs/evaluations/`. These are small development checks, not a calibrated false-positive or false-location rate.

## Independent pilot readiness

The multi-case post-fit evaluator is implemented. The empty pilot template returns `needs_input`; **no independent pilot recordings have been declared or evaluated**. Exact recorder-position evidence and a defensible mapping of audio sample zero to UTC are required before independent accuracy summaries. The current source-informed recording, approximate public camera marker and privacy-obfuscated AeroSonicDB coordinates do not supply that validation.

Public datasets were subsequently reviewed first, and a separate 40-recording acoustic screen was added. See the [public dataset audit and component check](public-datasets.md); none of its inputs has been promoted to an independently georeferenced location case.

For a small pilot, collect several new recordings, document microphone position and timing uncertainty separately, and freeze declared search bounds and settings before evaluating exact reference coordinates. Include quiet/unsuitable recordings and every failed or unresolved run. Each frozen case and separate reference file can then be listed in `pilot-manifest.json` and evaluated with `soundleak pilot-evaluate`.
