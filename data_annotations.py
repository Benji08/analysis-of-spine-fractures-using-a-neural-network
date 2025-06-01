import os
import re
import pandas as pd

INITIALS_COL_EXCEL = 'I.I.'
VERTEBRA_COL_EXCEL = 'Poziom'
FRACTURE_COL_MAP_EXCEL = {
    'A0': 'A0', 'A1': 'A1', 'A2': 'A2', 'A3': 'A3', 'A4': 'A4',
    'B1': 'B1', 'B2': 'B2', 'B3': 'B3', 'C': 'C'
}


def standardize_vertebra_id(id_str):
    """
    Standardizes a vertebra ID string to a common format.

    Examples:
    'Th1' -> 'T1'
    'L01' -> 'L1'
    't2'  -> 'T2'

    Args:
        id_str (str): The raw vertebra ID string.

    Returns:
        str or None: The standardized vertebra ID (e.g., 'T1', 'L5')
                     or None if the input cannot be parsed.
    """
    if not id_str or pd.isna(id_str):
        return None
    id_str = str(id_str).upper().replace("TH", "T")
    match = re.match(r'([TL])(\d+)', id_str)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return None


def parse_vertebra_filename(filename):
    """
    Parses a vertebra segmentation filename (e.g., 'L01.nrrd') to extract its components.

    Args:
        filename (str): The name of the .nrrd file.

    Returns:
        tuple or None: A tuple containing (vertebra_type, vertebra_number, original_filename, standardized_id)
                       (e.g., ('L', 1, 'L01.nrrd', 'L1')) if parsing is successful, otherwise None.
    """
    match = re.match(r'([TL])(\d+)\.nrrd', filename, re.IGNORECASE)
    if match:
        vert_type = match.group(1).upper()
        vert_num = int(match.group(2))
        standardized_id = f"{vert_type}{vert_num}"
        return vert_type, vert_num, filename, standardized_id
    return None


def get_anatomical_sort_key(parsed_info):
    """
    Generates a sort key for anatomically ordering vertebrae.

    Thoracic (T) vertebrae are sorted before Lumbar (L) vertebrae.
    Within each type, they are sorted by number (e.g., T1 < T2, L1 < L2).
    T12 < L1.

    Args:
        parsed_info (tuple): A tuple as returned by `parse_vertebra_filename`,
                             containing (vertebra_type, vertebra_number, ...).

    Returns:
        int: An integer key suitable for sorting. Lower values come first.
    """
    vert_type, vert_num, _, _ = parsed_info
    if vert_type == 'T':
        return vert_num
    elif vert_type == 'L':
        return vert_num + 100
    return 999


def load_and_parse_excel_data(excel_path, initials_col, vertebra_col, fracture_map_cols):
    """
    Loads and parses fracture data from an Excel file.

    The Excel file is expected to group rows by study, identified by entries
    in the `initials_col`. Each row can describe fractures for one or more
    vertebrae specified in `vertebra_col`. Fracture types are indicated by '1'
    in columns mapped by `fracture_map_cols`.

    Args:
        excel_path (str): Path to the Excel file.
        initials_col (str): Name of the Excel column identifying the start of a new study.
        vertebra_col (str): Name of the Excel column listing vertebra ID(s) (e.g., 'L1', 'T12/L1').
        fracture_map_cols (dict): A dictionary mapping Excel column names for fracture types
                                  to internal fracture codes (e.g., {'A0_ExcelCol': 'A0'}).

    Returns:
        dict: A dictionary where keys are study IDs (Excel row numbers stringified)
              and values are dictionaries. Each inner dictionary has standardized
              vertebra IDs as keys (e.g., 'L1') and values are dictionaries
              with 'A_type' and 'BC_type' fracture codes.
              Returns an empty dictionary if the file is not found or critical errors occur.
    """
    try:
        df = pd.read_excel(excel_path, dtype=str)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku Excel: {excel_path}")
        return {}
    except Exception as e:
        print(f"BŁĄD podczas wczytywania pliku Excel: {e}")
        return {}

    parsed_data = {}
    current_study_excel_row_id = None

    for index, row in df.iterrows():
        excel_actual_row_num_str = str(index + 2)

        try:
            initial_val = row[initials_col]
            vertebra_val = row[vertebra_col]
        except KeyError as e:
            print(f"BŁĄD: Brak oczekiwanej kolumny '{e}' w pliku Excel.")
            return {}

        if pd.notna(initial_val) and str(initial_val).strip() != "":
            current_study_excel_row_id = excel_actual_row_num_str
            parsed_data[current_study_excel_row_id] = {}

        if current_study_excel_row_id is None:
            continue

        raw_vertebra_desc = vertebra_val
        if pd.isna(raw_vertebra_desc) or str(raw_vertebra_desc).strip() == "":
            continue

        vertebra_ids_in_row = []
        parts = str(raw_vertebra_desc).split('/')
        for part in parts:
            std_id = standardize_vertebra_id(part.strip())
            if std_id:
                vertebra_ids_in_row.append(std_id)

        if not vertebra_ids_in_row:
            continue

        a_type = "__"
        bc_type = "__"

        for col_excel_name, fracture_code_internal in fracture_map_cols.items():
            if col_excel_name in row and pd.notna(row[col_excel_name]) and str(row[col_excel_name]).strip() == '1':
                if fracture_code_internal == 'C':
                    bc_type = 'C1'
                elif fracture_code_internal.startswith('A'):
                    a_type = fracture_code_internal
                elif fracture_code_internal.startswith('B'):
                    bc_type = fracture_code_internal

        for v_id in vertebra_ids_in_row:
            if v_id not in parsed_data[current_study_excel_row_id]:
                parsed_data[current_study_excel_row_id][v_id] = {'A_type': "__", 'BC_type': "__"}

            if a_type != "__":
                parsed_data[current_study_excel_row_id][v_id]['A_type'] = a_type
            if bc_type != "__":
                parsed_data[current_study_excel_row_id][v_id]['BC_type'] = bc_type
    return parsed_data


def process_study_folder(study_folder_path, study_excel_id, excel_fracture_data_for_study):
    """
    Processes a single study folder: deletes specified outermost vertebrae and renames
    the remaining .nrrd files to include fracture information.

    Deletion rules:
    - If the first vertebra is T1, the first 3 vertebrae (T1, T2, T3 if present) are protected
      from deletion at the superior end.
    - If the last vertebra is L5, the last 3 vertebrae (L5, L4, L3 if present) are protected
      from deletion at the inferior end.
    - Otherwise, up to two outermost vertebrae from each unprotected end are deleted,
      provided they are not part of an end protected by the other rule.

    Renaming:
    - Remaining .nrrd files are renamed to 'original_name-A_fracture-BC_fracture.nrrd'.
    - '__' is used if a fracture type is not specified.

    Args:
        study_folder_path (str): Path to the directory containing the study's .nrrd files.
        study_excel_id (str): The study ID (Excel row number) used to look up fracture data.
        excel_fracture_data_for_study (dict): A dictionary of fracture data for this specific study,
                                              as provided by `load_and_parse_excel_data`.
    """
    all_nrrd_files_info = []
    for filename in os.listdir(study_folder_path):
        if filename.lower().endswith(".nrrd"):
            parsed = parse_vertebra_filename(filename)
            if parsed:
                all_nrrd_files_info.append(parsed)

    if not all_nrrd_files_info:
        print("  Brak plików .nrrd w folderze.")
        return

    all_nrrd_files_info.sort(key=get_anatomical_sort_key)

    files_to_delete_names = set()
    do_not_delete_due_to_end_protection = set()

    low_end_is_T1_protected = False
    if all_nrrd_files_info:
        first_v_type, first_v_num, first_v_name, _ = all_nrrd_files_info[0]
        if first_v_type == 'T' and first_v_num == 1:
            low_end_is_T1_protected = True
            for i in range(min(3, len(all_nrrd_files_info))):
                do_not_delete_due_to_end_protection.add(all_nrrd_files_info[i][2])

    high_end_is_L5_protected = False
    if all_nrrd_files_info:
        last_v_type, last_v_num, last_v_name, _ = all_nrrd_files_info[-1]
        if last_v_type == 'L' and last_v_num == 5:
            high_end_is_L5_protected = True
            for i in range(min(3, len(all_nrrd_files_info))):
                do_not_delete_due_to_end_protection.add(all_nrrd_files_info[-(i + 1)][2])

    if not low_end_is_T1_protected and all_nrrd_files_info:
        deleted_count = 0
        for i in range(len(all_nrrd_files_info)):
            if deleted_count >= 2:
                break
            file_name_candidate = all_nrrd_files_info[i][2]
            if file_name_candidate not in do_not_delete_due_to_end_protection:
                files_to_delete_names.add(file_name_candidate)
                deleted_count += 1

    if not high_end_is_L5_protected and all_nrrd_files_info:
        deleted_count = 0
        for i in range(len(all_nrrd_files_info) - 1, -1, -1):
            if deleted_count >= 2:
                break
            file_name_candidate = all_nrrd_files_info[i][2]
            if file_name_candidate not in do_not_delete_due_to_end_protection:
                if file_name_candidate not in files_to_delete_names:
                    files_to_delete_names.add(file_name_candidate)
                    deleted_count += 1

    if files_to_delete_names:
        print(
            f"Skrajne pliki do usunięcia ({len(files_to_delete_names)}): {', '.join(sorted(list(files_to_delete_names)))}")
        for filename_to_delete in files_to_delete_names:
            file_path_to_delete = os.path.join(study_folder_path, filename_to_delete)
            try:
                os.remove(file_path_to_delete)
            except OSError as e:
                print(f"    BŁĄD podczas usuwania {filename_to_delete}: {e}")
    else:
        print("Brak skrajnych plików do usunięcia.")

    print("Zmiana nazw plików:")
    remaining_files_after_deletion_info = []
    for filename in os.listdir(study_folder_path):
        if filename.lower().endswith(".nrrd"):
            parsed = parse_vertebra_filename(filename)
            if parsed:
                remaining_files_after_deletion_info.append(parsed)

    if not remaining_files_after_deletion_info:
        print("  Brak plików .nrrd po usunięciu do zmiany nazwy.")
        return

    remaining_files_after_deletion_info.sort(key=get_anatomical_sort_key)

    for v_type, v_num, original_filename, standardized_id in remaining_files_after_deletion_info:
        fracture_details = excel_fracture_data_for_study.get(standardized_id, {'A_type': "__", 'BC_type': "__"})

        a_fracture = fracture_details['A_type']
        bc_fracture = fracture_details['BC_type']

        base_name, ext = os.path.splitext(original_filename)
        new_filename = f"{base_name}-{a_fracture}-{bc_fracture}{ext}"

        original_full_path = os.path.join(study_folder_path, original_filename)
        new_full_path = os.path.join(study_folder_path, new_filename)

        if original_filename != new_filename:
            try:
                os.rename(original_full_path, new_full_path)
                print(f"    Zmieniono nazwę: '{original_filename}' -> '{new_filename}'")
            except OSError as e:
                print(f"    BŁĄD zmiany nazwy '{original_filename}' na '{new_filename}': {e}")


def remove_and_rename_segmentations(main_folder_path, excel_file_path):
    """
    Main function to orchestrate the processing of vertebra segmentations.

    It iterates through subfolders in `main_folder_path`. For each subfolder,
    it extracts a study ID (Excel row number) from the folder name (expected format: "... xNUMBER").
    It then loads fracture data from `excel_file_path` and calls `process_study_folder`
    to delete specified vertebrae and rename remaining .nrrd files.


    Args:
        main_folder_path (str): Path to the main directory containing study subfolders.
        excel_file_path (str): Path to the Excel file with fracture data.
    """
    print("Rozpoczynanie skryptu...")
    excel_data_all_studies = load_and_parse_excel_data(
        excel_file_path,
        INITIALS_COL_EXCEL,
        VERTEBRA_COL_EXCEL,
        FRACTURE_COL_MAP_EXCEL
    )

    if not excel_data_all_studies:
        print("Nie udało się załadować danych z Excela lub plik jest pusty/błędny. Przerywam.")
        return

    if not os.path.isdir(main_folder_path):
        print(f"BŁĄD: Folder główny nie istnieje: {main_folder_path}")
        return

    processed_folders_count = 0
    for folder_name in os.listdir(main_folder_path):
        current_study_folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(current_study_folder_path):
            match = re.search(r' x(\d+)$', folder_name)
            if match:
                study_excel_id_from_folder = match.group(1)
                print(f"Folder: '{folder_name}', wyekstrahowano ID Excela: '{study_excel_id_from_folder}'")
            else:
                print(
                    f"Pominięto folder '{folder_name}': nie pasuje do wzorca nazwy z ID Excela.")
                continue

            fracture_data_for_this_study = excel_data_all_studies.get(study_excel_id_from_folder, {})

            if study_excel_id_from_folder not in excel_data_all_studies:
                print(
                    f"Ostrzeżenie: Brak danych w Excelu dla badania z ID Excela '{study_excel_id_from_folder}' (folder '{folder_name}'). Pliki otrzymają domyślny sufiks '-__-__'.")
            elif not fracture_data_for_this_study:
                print(
                    f"Informacja: Dla badania z ID Excela '{study_excel_id_from_folder}' (folder '{folder_name}') nie znaleziono informacji o złamaniach kręgów w danych Excela. Pliki otrzymają domyślny sufiks '-__-__'.")

            process_study_folder(current_study_folder_path, study_excel_id_from_folder, fracture_data_for_this_study)
            processed_folders_count += 1

    if processed_folders_count == 0:
        print("Nie znaleziono żadnych podfolderów pasujących do wzorca lub w folderze głównym do przetworzenia.")
    print("\nSkrypt zakończył działanie.")


if __name__ == "__main__":
    data_set_path = "C:/Users/barba/OneDrive/Pulpit/data_set"
    excel_path_main = "C:/Users/barba/OneDrive/Pulpit/Zanonimizowane_całość urazy kręgosłupa 2014-2020.xlsx"
    remove_and_rename_segmentations(data_set_path, excel_path_main)