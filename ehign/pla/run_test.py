# %%
import argparse
import json
import os
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
from EHIGN import DTIPredictor
from run_graph_constructor import GraphDataset, collate_fn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from utils import load_model_dict

# 不要在这里设置 CUDA_VISIBLE_DEVICES，使用外部传入的环境变量
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
warnings.filterwarnings("ignore")


# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, pr


def val_toy(model, dataloader, device, data_root):
    model.eval()
    results_dict = {}

    for data in dataloader:
        bg, label, file_path = data
        bg = bg.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2

            graph_list = dgl.unbatch(bg)
            for idx, g in enumerate(graph_list):
                file_path_str = file_path[idx]
                protein_id = file_path_str.split("/")[-2]
                # 提取配体名称（从文件名中解析）
                ligand_name = os.path.basename(file_path_str).replace("Graph_EHIGN-", "").replace(".dgl", "")
                
                if protein_id not in results_dict:
                    results_dict[protein_id] = {"protein_name": protein_id, "ligands": []}
                
                results_dict[protein_id]["ligands"].append({
                    "name": ligand_name,
                    "score": pred[idx].item()
                })

    external_df = pd.read_csv(os.path.join(data_root, "external_test.csv"))
    protein_order = external_df["pdbid"].unique().tolist()

    # 创建输出目录
    output_dir = "/data_lmdb/output"
    os.makedirs(output_dir, exist_ok=True)

    final_results = []

    print("\npredictions:")
    for protein_id in protein_order:
        if protein_id in results_dict:
            protein_data = results_dict[protein_id]
            print(f"{protein_id}: {len(protein_data['ligands'])} ligands")
            
            # 为每个蛋白生成单独的 JSON 文件
            output_file = os.path.join(output_dir, f"{protein_id}.json")
            with open(output_file, "w") as f:
                json.dump(protein_data, f, indent=2)
            print(f"  Saved to {output_file}")
            
            final_results.append(protein_data)

    print("\nthe final results is:")
    print(f"Generated {len(final_results)} protein prediction files")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data directory.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the model file"
    )
    args = parser.parse_args()
    data_root = args.data_dir
    data_dir = os.path.join(data_root, "external_test")
    data_df = pd.read_csv(os.path.join(data_root, "external_test.csv"))
    print(data_df)

    toy_set = GraphDataset(
        data_dir, data_df, graph_type="Graph_EHIGN", create=False
    )

    toy_loader = DataLoader(
        toy_set,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    device = torch.device("cuda:0")
    model = DTIPredictor(
        node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3
    ).to(device)
    model_path = args.model_dir
    model_position = os.path.join(model_path, "Ehignmodel.pt")
    load_model_dict(model, model_position)

    print("\nbegin to process datasets...")
    toy_predictions = val_toy(model, toy_loader, device, data_root)
    
    # 同时保存一个汇总的 JSON 文件（兼容旧代码）
    with open("/data_lmdb/toy_predictions.json", "w") as f:
        json.dump(toy_predictions, f, indent=2)
    print("\nalready saved to /data_lmdb/toy_predictions.json and /data_lmdb/output/<protein>.json")
