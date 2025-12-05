import argparse
import os
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_and_build_data, create_loader, train_model, evaluate
from models.rgcn import RGCN
from models.hgt import HGT
from models.han import HAN

# 双向日志记录器
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['RGCN', 'HAN', 'HGT'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 1. 日志设置
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{args.model}_{timestamp}.log")
    sys.stdout = Logger(log_filename)

    print(f"================================================")
    print(f"🚀 Start Training: {args.model}")
    print(f"📅 Time: {timestamp}")
    print(f"📂 Log saved to: {log_filename}")
    print(f"================================================")

    # 2. 路径配置
    BASE_DATA = "../data/benchmark"
    nodes_path = f"{BASE_DATA}/PrimeKG/nodes.csv"
    train_path = f"{BASE_DATA}/PrimeKG/train_edges.csv"
    val_path = f"{BASE_DATA}/PrimeKG/val_edges.csv"
    test_path = f"{BASE_DATA}/Kaggle_drug_repositioning/test.csv"
    test_hard_path = f"{BASE_DATA}/Kaggle_drug_repositioning/test_hard.csv"

    model_dir = f"../model/{args.model}"
    train_process_dir = f"../data/training_process/{args.model}"  # 【新增】训练过程目录
    eval_dir = f"../data/evaluation/{args.model}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(train_process_dir, exist_ok=True)  # 创建目录
    os.makedirs(eval_dir, exist_ok=True)

    config = {
        'max_epochs': 100,
        'patience': 10,
        'lr': 0.001,
        'best_model_path': os.path.join(model_dir, f"{args.model.lower()}_best.pt")
    }

    # 3. 加载数据
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path, test_hard_path
    )

    train_loader = create_loader(data, datasets['train'], 4096, [20, 10], shuffle=True)
    val_loader = create_loader(data, datasets['val'], 4096, [20, 10])
    test_loader = create_loader(data, datasets['test'], 4096, [20, 10])
    test_hard_loader = create_loader(data, datasets['test_hard'], 4096, [20, 10])

    # 4. 初始化模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == 'RGCN':
        model = RGCN(num_nodes, 128, num_rels).to(device)
        pass

    elif args.model == 'HGT':
        # 初始化 HGT，这里我们可以加一些特定参数比如 heads
        print("Initializing Heterogeneous Graph Transformer (HGT)...")
        model = HGT(
            num_nodes=num_nodes,
            hidden_dim=128,
            num_rels=num_rels,
            num_layers=2,
            num_heads=4  # HGT 特有参数
        ).to(device)
        pass

    elif args.model == 'HAN':
        print("Initializing Heterogeneous Graph Attention Network (HAN)...")
        # 注意：HAN 的显存占用很高，如果爆显存，请减小 hidden_dim 或 batch_size
        model = HAN(
            num_nodes=num_nodes,
            hidden_dim=128,
            num_rels=num_rels,
            num_layers=2,  # 虽然传了2，但上面的简单实现主要是单层聚合
            num_heads=4
        ).to(device)

    # 5. 训练逻辑
    if os.path.exists(config['best_model_path']):
        print(f"\nFound existing model: {config['best_model_path']}")
        print(">>> Skipping Training Phase...")
        model.load_state_dict(torch.load(config['best_model_path']))
    else:
        print("\nStarting Training Phase...")
        # 获取训练历史
        history = train_model(model, train_loader, val_loader, datasets['train'][2], datasets['val'][2], device, config)

        # 【新增】保存训练过程数据
        hist_df = pd.DataFrame(history)
        hist_csv_path = os.path.join(train_process_dir, "train_val_loss_auc.csv")
        hist_df.to_csv(hist_csv_path, index=False)
        print(f"Training history saved to: {hist_csv_path}")

        # 【新增】绘制 Loss 和 AUC 曲线
        plt.figure(figsize=(12, 5))

        # 子图 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss')
        plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()

        # 子图 2: AUC
        plt.subplot(1, 2, 2)
        plt.plot(hist_df['epoch'], hist_df['val_auc'], label='Val AUC', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Validation AUC')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(train_process_dir, "training_curves.svg"))
        plt.close()
        print(f"Training curves plot saved to: {train_process_dir}")

        # Load best for testing
        model.load_state_dict(torch.load(config['best_model_path']))

    # 6. 测试
    print("\n========== Evaluation: Standard Test Set ==========")
    evaluate(model, test_loader, datasets['test'][2], device, save_path=os.path.join(eval_dir, "standard"))

    if test_hard_loader:
        print("\n========== Evaluation: Hard Test Set (Degree Matched) ==========")
        evaluate(model, test_hard_loader, datasets['test_hard'][2], device, save_path=os.path.join(eval_dir, "hard"))


if __name__ == "__main__":
    main()