# Data

The dataset is **not included** in this repository. To run the notebook, place these two files in this folder:

| File       | Rows | Description |
| `train.csv`| 2,781| Labelled players (includes the `Drafted` column) |
| `test.csv` | 696  | Players without the `Drafted` label, scored to produce `outputs/predictions.csv` |

Expected columns: `Id, Year, Age, School, Height, Weight, Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Broad_Jump, Agility_3cone, Shuttle, Player_Type, Position_Type, Position` (plus `Drafted` in `train.csv`). See the main README for units and descriptions.

The data was provided to me through a data science bootcamp, so I do not redistribute it here.

**Keeping the files somewhere else?** Set the `NFL_DATA_DIR` environment variable to the folder that contains them, e.g.

```bash
# macOS / Linux
export NFL_DATA_DIR=/path/to/folder
# Windows (PowerShell)
$env:NFL_DATA_DIR = "C:\path\to\folder"
```
