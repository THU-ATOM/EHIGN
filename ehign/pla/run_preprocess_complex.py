import argparse
import os
import pickle
import re

import pandas as pd
import pymol
from rdkit import Chem, RDLogger
from tqdm import tqdm
from openbabel import pybel
RDLogger.DisableLog("rdApp.*")

def sdf_to_mol2(sdf_str):
    """After using RDKit's Sanitize to correct the structure, convert it to MOL2
    through pybel."""
    rdkit_mol = Chem.MolFromMolBlock(sdf_str, sanitize=False)
    if rdkit_mol is None:
        raise ValueError("RDKit cannot parse SDF content")

    try:
        Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except Exception as e:
        print(f"Structural correction failed: {str(e)}")

        rdkit_mol = Chem.AddHs(rdkit_mol, addCoords=True)
    corrected_sdf = Chem.MolToMolBlock(rdkit_mol)

    mol = pybel.readstring("sdf", corrected_sdf)
    return mol.write("mol2")

def generate_mol2_from_sdf(sdf_file, protein_name, ligand_name, workdir):
    """Convert SDF files to MOL2 files according to the file structure
    requirements of this project."""

    data_dir = os.path.join(workdir, "data", "external_test", protein_name)
    os.makedirs(data_dir, exist_ok=True)

    mol2_filename = os.path.join(data_dir, f"{protein_name}_{ligand_name}.mol2")

    try:

        with open(sdf_file, "r") as f:
            sdf_str = f.read()

        mol2_str = sdf_to_mol2(sdf_str)

        with open(mol2_filename, "w") as f:
            f.write(mol2_str)

        print(f"MOL2 file has been generated：{mol2_filename}")

    except Exception as e:
        print(f"Error converting SDF to MOL2: {str(e)}")
        raise


def generate_pocket(data_dir, distance=5):
    complex_ids = os.listdir(data_dir)
    for cid in complex_ids:
        print(cid)
        complex_dir = os.path.join(data_dir, cid)
        protein_path = os.path.join(complex_dir, f"{cid}_protein.pdb")

        for file in os.listdir(complex_dir):
            if file.endswith(".mol2") and file != f"{cid}_protein.mol2":

                ligand_name = os.path.splitext(file)[0].replace(" ", "_")

                ligand_name = re.sub(r"[^a-zA-Z0-9_-]", "", ligand_name)
                pocket_path = os.path.join(
                    complex_dir, f"{cid}_{ligand_name}_pocket_{distance}A.pdb"
                )

                if os.path.exists(pocket_path):
                    continue

                pymol.cmd.load(protein_path)
                pymol.cmd.remove("resn HOH")
                lig_native_path = os.path.join(complex_dir, file)
                pymol.cmd.load(lig_native_path, object=ligand_name)

                objects = pymol.cmd.get_object_list()
                if ligand_name not in objects:
                    print(
                        f"Warning: Ligand {ligand_name} not loaded correctly."
                    )
                    continue

                pymol.cmd.remove("hydrogens")
                try:
                    pymol.cmd.select(
                        "Pocket", f"byres {ligand_name} around {distance}"
                    )
                    pymol.cmd.save(pocket_path, "Pocket")
                except pymol.CmdException as e:
                    print(f"Error selecting pocket for {ligand_name}: {e}")
                pymol.cmd.delete("all")


def generate_complex(data_dir, data_df, distance=5, input_ligand_format="mol2"):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pKa = row["pdbid"], float(row["-logKd/Ki"])
        complex_dir = os.path.join(data_dir, cid)

        for file in os.listdir(complex_dir):
            if (
                file.endswith(f".{input_ligand_format}")
                and file != f"{cid}_protein.{input_ligand_format}"
            ):

                ligand_name = os.path.splitext(file)[0].replace(" ", "_")
                # avoid naming errors
                ligand_name = re.sub(r"[^a-zA-Z0-9_-]", "", ligand_name)
                pocket_path = os.path.join(
                    complex_dir, f"{cid}_{ligand_name}_pocket_{distance}A.pdb"
                )

                # 统一使用你提供的 generate_mol2_from_sdf 完成 SDF->MOL2，再由后续流程生成 PDB/口袋
                ligand_input_path = os.path.join(complex_dir, file)
                try:
                    # 当输入为 SDF（推荐）：直接调用生成 MOL2，输出会写入 workdir/data/external_test/<cid>/
                    if file.lower().endswith(".sdf"):
                        generate_mol2_from_sdf(
                            sdf_file=ligand_input_path,
                            protein_name=cid,
                            ligand_name=ligand_name,
                            workdir=data_root,
                        )
                    else:
                        # 若不是 SDF（例如 mol2、pdb），优先保持原路径作为 ligand_path，供后续 RDKit 读取
                        pass
                except Exception as e:
                    print(f"Error converting {ligand_name} via generate_mol2_from_sdf: {e}")
                # 兼容后续使用：如果存在 PDB 输入则保留其路径；否则尝试从同目录中找到已生成的 PDB
                if file.lower().endswith(".pdb"):
                    ligand_path = os.path.join(complex_dir, file)
                else:
                    ligand_path = os.path.join(complex_dir, file.replace(f".{input_ligand_format}", ".pdb"))

                save_path = os.path.join(
                    complex_dir, f"{cid}_{ligand_name}.rdkit"
                )

                if not os.path.exists(ligand_path):
                    print(f"Warning: Ligand file {ligand_path} does not exist.")
                    continue

                ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
                if ligand is None:
                    print(
                        f"Unable to process ligand of {ligand_name} from {ligand_path}"
                    )
                    continue

                if not os.path.exists(pocket_path):
                    print(f"Warning: Pocket file {pocket_path} does not exist.")
                    continue

                pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
                if pocket is None:
                    print(
                        f"Unable to process protein of {cid} from {pocket_path}"
                    )
                    continue

                complex = (ligand, pocket)
                with open(save_path, "wb") as f:
                    pickle.dump(complex, f)

        pbar.update(1)


def remove_negative_oxygen_notation(file_path):
    """Remove negative charges of O."""
    with open(file_path, "r") as f:
        content = f.read()

    modified_content = re.sub(r"O1-", "O", content)

    with open(file_path, "w") as f:
        f.write(modified_content)


def process_files_in_directory(data_dir):
    """process 5a files."""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("5A.pdb"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                remove_negative_oxygen_notation(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data directory.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    args = parser.parse_args()

    distance = 5
    input_ligand_format = "mol2"
    data_root = args.data_dir
    data_dir = os.path.join(data_root, "external_test")
    data_df = pd.read_csv(os.path.join(data_root, "external_test.csv"))
   
    generate_pocket(data_dir=data_dir, distance=distance)
    process_files_in_directory(data_dir)
    generate_complex(
        data_dir,
        data_df,
        distance=distance,
        input_ligand_format=input_ligand_format,
    )
