# Label audit

Labelling rule from the assignment: each photo goes to the album of its **main subject**
(Food & Drinks, People or Interior). All audits below were done by one annotator, the AI coding assistant
that prepared the data, from visual inspection with no second reviewer. Every file move is also reflected in
the `class` column of `manifest.csv`.

## 1. Training `Interior` (all 50 images audited)

All 50 `train/interior` images came from Places365 *coffee_shop*, which labels the **scene**, not the
subject. 23 were relabelled:

| File | New label | Main subject |
|---|---|---|
| `0511f55b1f784c959adc8dc7762539e8.jpg` | people | man posing at a counter |
| `09bc59d36eb64c119e8e69181d35af2e.jpg` | people | crowd writing at a tasting counter |
| `4439ee9db8aa4635996d9e79abcd6254.jpg` | people | portrait of a man |
| `455a54336d5d4ced851941860f13f70a.jpg` | people | family at a table |
| `5fcd7635521f473994040cbce2a2dad6.jpg` | people | people at a window table |
| `7f0d56c03b2e499ba4fe18ad863f83ce.jpg` | people | man carrying a plate |
| `8e2a5de08edc44dd9a555717a829f296.jpg` | people | customers served at a table |
| `c6ec9dbd41f64f33bb7ea33a7b546d5c.jpg` | people | woman seated, posing |
| `d1b199e8d0804f9385c077b5044dd073.jpg` | people | group at tables |
| `de090d15c49f442aadb51be8941dd8bb.jpg` | people | couple at a table |
| `e70116788d494c7680fee651700638d5.jpg` | people | group at a table |
| `e9f9cc6b02c84738a11c8c37160703ea.jpg` | people | three children at a table |
| `f0797d131ffc4d74884b03174e966d7e.jpg` | people | barista making syphon coffee |
| `13d38b417199480b93546b971cc1626f.jpg` | food_drinks | milk steaming at an espresso machine (ambiguous: equipment + hands) |
| `27b610fb51eb4d6da6950d1e9fb57b73.jpg` | food_drinks | coffee cup close-up |
| `4a7765aa2b41467483bf50a7c40e1d22.jpg` | food_drinks | takeaway cup and bag on a table |
| `6830ce31a03840b4be912218422ba55c.jpg` | food_drinks | coffee cupping: cups and beans |
| `6caf691271d24a1a9bfed24fe5582305.jpg` | food_drinks | stacked cups (ambiguous: tableware) |
| `bc524cc033d14ed2906a8187ae0ff6bb.jpg` | food_drinks | cakes and cups on a table |
| `d326b5727c214d29a92dfb20d532cf0c.jpg` | food_drinks | espresso cup close-up |
| `dbcfd8b26713499687eceefa62f5db99.jpg` | food_drinks | espresso machine with cups at a stand (ambiguous: equipment) |
| `e74e0149085543c8b0bc568627927880.jpg` | food_drinks | sandwiches and cans on a table |
| `f7de5533405d4356929034f5a10ea9ca.jpg` | food_drinks | plate of cake, woman's torso behind (ambiguous) |

**Ambiguous, kept as Interior (27 remain):**
- 11 exterior or terrace shots of a café: `643459311a3345a29db307aa502948c6.jpg`, `6a981bcc1fb547578ad5a862376658e3.jpg`, `735859745b3f449c8690c9b04c680cf9.jpg`, `7eea99f61c104e16ac1d1ebe484f0ffc.jpg`, `8b031c5c6cf0473daa988a8e486f769b.jpg`, `a2e37f2d6fd448bbb1441e0c3437604f.jpg`, `b4ce8fc682d94cfa822fbe9f9e4ae100.jpg`, `bb3a4870b9254b1fb038e6d405e464a9.jpg`, `c85b261a4d7b4827a5cf611ba4e64dff.jpg`, `d60648b7cc024b9aa93b78a1d4aff188.jpg`, `f393199dcbf943aa9bba94f549eb6b58.jpg`. No album fits better, and a café owner
  would most likely file the venue itself under Interior.
- `02a8cf899cd24ced87597aa216a7d492.jpg` and `826997e4cf6e47eeae2c4137eca5ee22.jpg` show staff behind the counter, but the room dominates.

## 2. Benchmark test split (`data/test`)

| File | Old | New | Main subject | Audited |
|---|---|---|---|---|
| `337d124eeda7482b9aba84aae4ef9528.jpg` | interior | food_drinks | pastry display case filling the frame | round 1 |
| `02efbaea0f1c4e96b6ffbb1f6a19d919.jpg` | interior | people | two people in aprons fill the frame | round 2 |
| `2fb066309ba54573a5d8f03d4738f692.jpg` | interior | people | man posing centre-frame | round 2 |
| `381278a11557485ab28fb61aa27c1951.jpg` | interior | people | four seated people, close up | round 2 |
| `92c536eaccbb4084bdc3e20c52375b11.jpg` | interior | food_drinks | latte-art close-up | round 2 |

**Ambiguous, kept as Interior:** `0ae7783d3e9d4de3b6246e9b9d2fce0a.jpg` (two boys seated, but the room and
doorway dominate), `5432e6aea9e543e5a805bf90d9887feb.jpg` and `bcfaa5e5…` (staff behind a counter), and
the exterior/terrace shots `807515e8…` and `b630edee…`.

**Caveat on round 2.** It audited all 14 remaining test Interior images by the rule above, but it happened
*after* the retrained model's predictions on the old labels were seen (the retrained model disagreed on 5).
Every image was judged by the same rule and 0ae7783d was kept as Interior although the model calls it People,
but the timing is a source of bias. The README therefore reports test scores with both the original and the
audited labels. The People and Food & Drinks test images (COCO, Food-101/Wikimedia) were not individually audited.

## 3. Café photos (`holdout_cafe/`, `holdout_fresh/`, café rows of `manifest.csv`)

Labels were assigned by the rule from the images alone, **before any prediction was run** on them (timestamps in
`labeled_at_utc`). Photos with a prominent second subject are marked `mixed_subject=yes`, and `label_reason`
explains the choice. To re-evaluate with your own labels, move files between class folders and re-run
`photo-sorter evaluate`.
