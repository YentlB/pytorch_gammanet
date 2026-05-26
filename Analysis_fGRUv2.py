from pathlib import Path
import nbformat
from textwrap import dedent

nb_path = "/mnt/data/gammanet_attention_analysis_notebook.ipynb"
nb = nbformat.read(nb_path, as_version=4)

# Insert/update a dataset parsing section
new_cell = nbformat.v4.new_code_cell(dedent("""
# ============================================================
# AUTOMATISCHE DATASET PARSING
# ============================================================

import re
from collections import defaultdict

IMAGE_ROOT = '/home/yentl/pytorch_gammanet/Images'

pattern = re.compile(
    r'(straight|C)_(high|low)_(TR|TL|BR|BL)_(\\d)_J(\\d{3})_(\\d+)'
)

dataset_index = []

for fname in os.listdir(IMAGE_ROOT):

    match = pattern.match(fname)

    if match is None:
        continue

    contour_type = match.group(1)
    contrast = match.group(2)
    quadrant = match.group(3)
    contour_position = int(match.group(4))
    jitter = int(match.group(5))
    stimulus_id = int(match.group(6))

    dataset_index.append({
        'filename': fname,
        'path': os.path.join(IMAGE_ROOT, fname),
        'contour_type': contour_type,
        'contrast': contrast,
        'quadrant': quadrant,
        'position': contour_position,
        'jitter': jitter,
        'stimulus_id': stimulus_id,
    })

dataset_df = pd.DataFrame(dataset_index)

print(dataset_df.head())
print()
print(f'Total images: {len(dataset_df)}')
"""))

# Replace old DATA_CONFIG section if found
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "DATA_CONFIG" in cell.source:
        nb.cells[i] = new_cell
        break

# Replace analyze_condition function
replacement = dedent("""
def analyze_condition(
    model,
    contour_type='straight',
    contrast='high',
    jitter_level=0,
):

    subset = dataset_df[
        (dataset_df['contour_type'] == contour_type) &
        (dataset_df['contrast'] == contrast) &
        (dataset_df['jitter'] == jitter_level)
    ]

    results = []

    for _, row in tqdm(subset.iterrows(), total=len(subset)):

        model.reset_hidden_states()

        img = Image.open(row['path']).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)

        sample_result = {
            'filename': row['filename'],
            'quadrant': row['quadrant'],
            'position': row['position'],
            'jitter': row['jitter'],
            'contrast': row['contrast'],
            'contour_type': row['contour_type'],
            'output_mean': output.mean().item(),
        }

        for layer_name in [
            'h0_exc',
            'h1_exc',
            'h2_exc',
            'h3_exc',
            'h4_exc',
        ]:

            tensor = getattr(model, layer_name)

            if tensor is not None:

                sample_result[layer_name] = global_channel_response(
                    tensor.detach().cpu()
                )

        results.append(sample_result)

    return results
""")

for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "def analyze_condition" in cell.source:
        nb.cells[i] = nbformat.v4.new_code_cell(replacement)
        break

# Add interpretation section for metadata
interpretation_cell = nbformat.v4.new_markdown_cell("""
# Waarom deze metadata belangrijk is

Je stimulusnamen bevatten experimenteel extreem waardevolle informatie:

- contour_type → straight vs C
- contrast → high vs low
- quadrant → ruimtelijke aandacht / locatie-effecten
- position → variatie binnen stimulusklasse
- jitter → psychofysische moeilijkheidsgraad

Daardoor kun je uiteindelijk analyses doen zoals:

- Welke populaties coderen contourtype?
- Welke populaties zijn contrastgevoelig?
- Welke recurrente states blijven stabiel bij hoge jitter?
- Werkt attention sterker voor low contrast?
- Zijn top-down effecten locatie-afhankelijk?
- Verschilt recurrente persistentie tussen C en straight contours?

Dit maakt de uiteindelijke paper veel sterker.
""")

nb.cells.append(interpretation_cell)

updated_path = "/mnt/data/gammanet_attention_analysis_notebook_v2.ipynb"

with open(updated_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(updated_path)
