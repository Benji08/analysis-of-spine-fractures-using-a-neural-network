import pandas as pd
import os
import pydicom

def extract_initials_from_dicom(dicom_folder):
    for root, _, files in os.walk(dicom_folder):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)

                # Pobierz nazwę pacjenta
                name = dcm.get("PatientName", None)
                if name:
                    if isinstance(name, pydicom.valuerep.PersonName):
                        initials = f"{name.family_name[0].upper()}.{name.given_name[0].upper()}."
                    else:
                        parts = str(name).split()
                        initials = ''.join(part[0].upper() + '.' for part in parts)
                else:
                    initials = None

                # Pobierz datę badania
                study_date = dcm.get("StudyDate", None)  # np. "20230522"
                if study_date:
                    formatted_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
                else:
                    formatted_date = None

                if initials and formatted_date:
                    return initials, formatted_date

            except Exception:
                continue
    return None, None

def extract_fractured_vertebrae(df, initials):
    patient_rows = df[df.iloc[:, 0] == initials]
    if patient_rows.empty:
        return {}

    fractures = {}
    for _, row in patient_rows.iterrows():
        vertebra = row['Poziom']
        for col in df.columns[2:]:
            value = row[col]
            if pd.notna(value):
                vertebra_code = vertebra.upper().replace(' ', '')  # np. 'L1'
                if len(vertebra_code) == 2:
                    vertebra_code = vertebra_code[0] + '0' + vertebra_code[1]
                if vertebra_code not in fractures:
                    fractures[vertebra_code] = []
                fractures[vertebra_code].append(col)
    return fractures

def rename_segmented_files(segmentation_root, fractures):
    for folder in os.listdir(segmentation_root):
        folder_path = os.path.join(segmentation_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith('.nrrd'):
                continue

            for vertebra, types in fractures.items():
                if file.startswith(vertebra):
                    old_path = os.path.join(folder_path, file)
                    new_name = f"{vertebra}_{'_'.join(types)}.nrrd"
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

def process_segmentations(dicom_root, segmentation_root, excel_path):
    df = pd.read_excel(excel_path)

    for patient_folder in os.listdir(dicom_root):
        patient_path = os.path.join(dicom_root, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        initials = extract_initials_from_dicom(patient_path)
        if not initials:
            print(f"[WARN] Nie znaleziono inicjałów dla folderu: {patient_folder}")
            continue

        fractures = extract_fractured_vertebrae(df, initials)
        if not fractures:
            print(f"[INFO] Brak złamań dla pacjenta: {initials}")
            continue

        rename_segmented_files(segmentation_root, fractures)
