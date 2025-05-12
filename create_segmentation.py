from pathlib import Path
from extract_bboxes_from_segmentation import read_and_extract


base_dir = Path(__file__).parent
badania_dir = base_dir / "serie"
segmentacje_dir = base_dir / "segmentacje"

# Znajdź wszystkie pliki kończące się na 'segmentation.nrrd'
for seg_path in badania_dir.glob("* segmentation.nrrd"):
    base_name = seg_path.stem.replace(" segmentation", "")
    image_path = badania_dir / f"{base_name}.nrrd"
    print(f"Przetwarzam badanie: {base_name}")

    if not image_path.exists():
        print(f"Brak pliku: {image_path.name}")
        continue

    output_dir = segmentacje_dir / base_name
    read_and_extract(str(seg_path), str(image_path), str(output_dir))
