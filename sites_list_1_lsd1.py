import os
import csv
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa

# ========== 配置区（只改这里） ==========
# 1. DSSP工具路径（改成你解压后的mkdssp.exe路径）
os.environ["DSSP_EXEC"] = r"d:\pythontools\DSSP\bin\mkdssp.exe"
# 2. 你的文件路径
PDB_DIR = "pdb_files_lsd1"          # PDB文件夹
SITE_LIST_FILE = "sites_list_lsd1.csv"  # 位点列表
OUTPUT_FILE = "lsd1_features_with_dssp.csv"  # 输出文件
WINDOW = 5   # ±5氨基酸窗口
# =======================================

# 提取pLDDT值
def build_plddt_index(model):
    plddt_index = {}
    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            if is_aa(residue) and "CA" in residue:
                resseq = residue.get_id()[1]
                plddt = residue["CA"].get_bfactor()
                plddt_index[(chain_id, resseq)] = plddt
    return plddt_index

# 提取DSSP特征（AA、SS、RSA、Phi、Psi）
def build_dssp_index(dssp):
    dssp_index = {}
    for key in dssp.keys():
        chain_id = key[0]
        resseq = key[1][1]
        # DSSP返回的元组：(AA, SS, RSA, Phi, Psi, ...)
        aa = dssp[key][0]
        ss = dssp[key][2]
        rsa = dssp[key][3]
        phi = dssp[key][4]
        psi = dssp[key][5]
        dssp_index[(chain_id, resseq)] = (aa, ss, rsa, phi, psi)
    return dssp_index

# 提取窗口内所有特征
def extract_window_features(protein_id, chain_id, center_res, label, dssp_index, plddt_index, window=5):
    row = {
        "Protein_ID": protein_id,
        "Chain": chain_id,
        "Site": center_res,
        "Label": label
    }
    # 遍历±5窗口
    for offset in range(-window, window+1):
        target_res = center_res + offset
        key = (chain_id, target_res)
        # 填充DSSP特征
        if key in dssp_index:
            aa, ss, rsa, phi, psi = dssp_index[key]
        else:
            aa, ss, rsa, phi, psi = "X", "-", "", "", ""
        # 填充pLDDT特征
        plddt = plddt_index.get(key, "")
        # 写入行
        row[f"AA_{offset}"] = aa
        row[f"pLDDT_{offset}"] = plddt
        row[f"SS_{offset}"] = ss
        row[f"RSA_{offset}"] = rsa
        row[f"Phi_{offset}"] = phi
        row[f"Psi_{offset}"] = psi
    return row

# 主函数
def main():
    # 读取位点列表
    sites = []
    with open(SITE_LIST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sites.append({
                "protein_id": row["Protein_ID"],
                "pdb_file": row["PDB_file"],
                "chain": row.get("Chain", "A"),
                "residue": int(row["Residue"]),
                "label": int(row["Label"])
            })
    
    # 按PDB文件分组（避免重复解析）
    pdb_groups = defaultdict(list)
    for site in sites:
        pdb_groups[site["pdb_file"]].append(site)
    
    all_features = []
    parser = PDBParser(QUIET=True)  # 关闭警告
    
    # 处理每个PDB文件
    for pdb_file, site_group in pdb_groups.items():
        pdb_path = os.path.join(PDB_DIR, pdb_file)
        # 检查PDB文件是否存在
        if not os.path.exists(pdb_path):
            print(f"警告：找不到PDB文件 {pdb_path}，跳过")
            continue
        
        # 解析PDB + 计算DSSP
        structure = parser.get_structure(site_group[0]["protein_id"], pdb_path)
        model = structure[0]
        try:
            dssp = DSSP(model, pdb_path, dssp=os.environ["DSSP_EXEC"])
        except Exception as e:
            print(f"计算DSSP失败 {pdb_file}：{e}")
            continue
        
        # 构建索引
        plddt_index = build_plddt_index(model)
        dssp_index = build_dssp_index(dssp)
        
        # 提取每个位点的特征
        for site in site_group:
            feat = extract_window_features(
                site["protein_id"],
                site["chain"],
                site["residue"],
                site["label"],
                dssp_index,
                plddt_index,
                WINDOW
            )
            all_features.append(feat)
    
    # 保存结果
    if all_features:
        headers = all_features[0].keys()
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_features)
        print(f"成功！结果保存到 {OUTPUT_FILE}")
    else:
        print("没有提取到特征，请检查PDB文件和DSSP路径")

if __name__ == "__main__":
    main()