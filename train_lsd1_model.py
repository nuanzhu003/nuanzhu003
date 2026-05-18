# 这个是采用± 5进行的分析，有些过拟合
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt # 这两行用于可视化
from sklearn.metrics import roc_curve # 这两行用于可视化

# =========================
# 1. 读入你的 TSV 特征文件
# =========================

FILE = "lsd1_features_all_sites.csv"   # 你的文件名（即便是 TSV，也可以叫 .csv）

df = pd.read_csv(FILE)   # ✅ 注意：这里必须是 sep="\t"，机器自动生成的文件反倒是正常的
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].astype(int)
print("\n清理后的 Label 分布：")
print(df["Label"].value_counts())


print("数据维度：", df.shape)
print("前10列列名：")
print(df.columns.tolist()[:10])

print("\nLabel 分布：")
print(df["Label"].value_counts())


# =========================
# 2. 拆分特征 X 和标签 y
# =========================

id_cols = ["Protein_ID", "Chain", "Site"]
label_col = "Label"

feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

X = df[feature_cols]
y = df[label_col]


# =========================
# 3. 分类特征 vs 数值特征
# =========================

# 氨基酸 + 二级结构 → 分类特征
cat_cols = [c for c in feature_cols if c.startswith("AA_") or c.startswith("SS_")]

# 其余 → 数值特征
num_cols = [c for c in feature_cols if c not in cat_cols]

print("\n分类特征数量：", len(cat_cols))
print("数值特征数量：", len(num_cols))


# =========================
# 4. 训练 / 测试集划分
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. 预处理 + 随机森林模型
# =========================

preprocess = ColumnTransformer(
    transformers=[
        # 分类特征：先用众数填补，再做 one-hot
        ("cat", 
         Pipeline(steps=[
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("onehot", OneHotEncoder(handle_unknown="ignore"))
         ]),
         cat_cols),
        
        # 数值特征：先用中位数填补，再标准化
        ("num", 
         Pipeline(steps=[
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         num_cols),
    ]
)


clf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf),
])


# =========================
# 6. 训练模型
# =========================

model.fit(X_train, y_train)
# ======== 训练集 AUC（判断是否过拟合）========
y_train_proba = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, y_train_proba)
print("训练集 ROC-AUC:", train_auc)


# =========================
# 7. 测试集评估
# =========================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n================ 分类报告 ================")
print(classification_report(y_test, y_pred))

print("=============== ROC-AUC ===============")
print(roc_auc_score(y_test, y_proba))

# =========================
# 8. 绘制并保存 ROC 曲线（sequence-only 模型）
# =========================

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (window5 Model, AUC = {roc_auc:.3f})")
plt.tight_layout()

# ✅ 自动保存为论文用图片
plt.savefig("ROC_win5_lsd1.png", dpi=300)
plt.close()

print("✅ 已生成并保存 ROC 图像：ROC_win5.png")
