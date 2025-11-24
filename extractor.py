
import argparse
import os
import sys
import time
import numpy as np
import torch
import gorilla
import spconv.pytorch as spconv
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from torch_scatter import scatter_mean, scatter_max

# ==========================================
# 关键设置：优先导入本地编译的 pointgroup_ops
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'spformer', 'lib')

# 将 lib 路径插入到 sys.path 的最前面
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
    print(f"[Info] 已添加本地库路径到 sys.path: {lib_path}")

# 将当前目录加入 path，以便能导入 spformer 包
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import pointgroup_ops
    print(f"[Info] 成功导入 pointgroup_ops 从: {pointgroup_ops.__file__}")
except ImportError as e:
    print(f"[Error] 无法导入 pointgroup_ops。请确保已在 {lib_path} 下运行 setup.py build_ext --inplace")
    raise e

from spformer.model import SPFormer
from spformer.utils import get_root_logger


class FeatureExtractor:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        初始化特征提取器
        Args:
            config_path: 配置文件路径
            checkpoint_path: 预训练权重路径
            device: 运行设备
        """
        self.device = device
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        self.cfg = gorilla.Config.fromfile(config_path)
        self.logger = get_root_logger()
        
        # 构建模型
        print(f"[Info] 正在构建模型 SPFormer...")
        self.model = SPFormer(**self.cfg.model).to(device)
        
        # 加载权重
        print(f'[Info] 从 {checkpoint_path} 加载权重...')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"权重文件未找到: {checkpoint_path}")
            
        gorilla.load_checkpoint(self.model, checkpoint_path, strict=False)
        self.model.eval()

    def load_data(self, file_path):
        """
        加载S3DIS格式的点云数据 (x, y, z, r, g, b)
        """
        print(f"[Info] 正在加载数据: {file_path}")
        data = np.loadtxt(file_path)
        xyz = data[:, :3]
        rgb = data[:, 3:6]
        # 归一化RGB到 [-1, 1] (ScanNet 标准)
        rgb = rgb / 127.5 - 1.0
        return xyz, rgb

    def generate_superpoints(self, xyz, rgb, n_clusters=512):
        """
        使用KNN和层次聚类生成几何连续的超点，模拟segmentator的行为
        """
        print(f"[Info] 正在生成超点 (目标数量: {n_clusters})...")
        t0 = time.time()
        
        n_points = len(xyz)
        # 为了加速图构建，如果点数过多则进行降采样构建图
        # 注意：这只是为了加速图构建，最终标签会传播回所有点
        if n_points > 30000:
            print(f"  - 输入点数 {n_points} > 30000，进行降采样以加速图构建...")
            # 使用随机采样
            idx = np.random.choice(n_points, 30000, replace=False)
            xyz_sub = xyz[idx]
            rgb_sub = rgb[idx]
        else:
            idx = np.arange(n_points)
            xyz_sub = xyz
            rgb_sub = rgb

        # 构建KNN图 (保证几何邻近性)
        print("  - 构建KNN连接图...")
        # n_neighbors=10 既能保证连通性，又能限制边的数量
        connectivity = kneighbors_graph(xyz_sub, n_neighbors=10, include_self=False)

        # 执行层次聚类
        print("  - 执行层次聚类 (Agglomerative Clustering)...")
        # ward 链接方式倾向于生成大小均匀的簇
        cluster = AgglomerativeClustering(
            n_clusters=n_clusters, 
            connectivity=connectivity, 
            linkage='ward'
        )
        # 使用XYZ和RGB作为特征进行聚类 (权重: XYZ=1.0, RGB=1.0)
        # 可以调整权重来更关注空间位置或颜色
        labels_sub = cluster.fit_predict(np.concatenate([xyz_sub, rgb_sub], axis=1))

        # 如果进行了降采样，需要将标签传播回所有点
        if n_points > 30000:
            print("  - 将超点标签传播回原始点云...")
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(xyz_sub, labels_sub)
            # 对所有点预测所属超点
            labels = knn.predict(xyz)
        else:
            labels = labels_sub

        unique_labels = len(np.unique(labels))
        print(f"[Info] 超点生成完成，耗时 {time.time()-t0:.2f}秒。唯一超点数量: {unique_labels}")
        return labels

    def prepare_batch(self, xyz, rgb, superpoints):
        """
        准备模型输入的batch数据，包括体素化
        """
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoints).long()
        
        # 计算Batch偏移量 (对于单样本Batch，结构为 [0, max_sp_id + 1])
        batch_offsets = [0, superpoint.max().item() + 1]
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        
        # 获取体素配置
        voxel_cfg = self.cfg.data.test.voxel_cfg
        scale = voxel_cfg.scale
        
        # 应用缩放并将坐标移动到正数区域以便体素化
        # ScanNetV2 数据处理逻辑: xyz_middle * scale
        coord_float_scaled = coord_float * scale
        coord_float_scaled -= coord_float_scaled.min(0)[0]
        
        # 添加Batch索引 (维度0)
        coord_long = coord_float_scaled.long()
        # [N, 4] -> (batch_idx, x, y, z)
        coords_with_batch = torch.cat([torch.LongTensor(coord_long.shape[0], 1).fill_(0), coord_long], 1)
        
        # 拼接RGB和浮点坐标作为特征 [N, C+3]
        feats = torch.cat((feat, coord_float_scaled), dim=1)
        
        # 使用pointgroup_ops进行体素化
        # 计算空间形状并裁剪，防止越界
        spatial_shape_clip = np.clip((coords_with_batch.max(0)[0][1:] + 1).numpy(), voxel_cfg.spatial_shape[0], None)
        
        # 调用 C++ 扩展进行体素化索引计算
        # voxel_coords: [M, 4] (batch_idx, z, y, x) 注意顺序
        # p2v_map: [N] 点到体素的映射
        # v2p_map: [M] 体素到点的映射 (用于后续聚合)
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords_with_batch, 1, 4)
        
        return {
            'scan_ids': ["demo_scan"],
            'voxel_coords': voxel_coords.to(self.device),
            'p2v_map': p2v_map.to(self.device),
            'v2p_map': v2p_map.to(self.device),
            'spatial_shape': spatial_shape_clip,
            'feats': feats.to(self.device),
            'superpoints': superpoint.to(self.device),
            'batch_offsets': batch_offsets.to(self.device)
        }

    def extract(self, xyz, rgb, superpoints):
        """
        执行特征提取流程
        """
        batch = self.prepare_batch(xyz, rgb, superpoints)
        
        print("\n" + "="*50)
        print(" 开始特征提取流程 ")
        print("="*50)
        
        with torch.no_grad():
            # --- 步骤 1: 体素化输入 ---
            print(f"\n[步骤 1] 体素化输入:")
            print(f"  - 点特征形状: {batch['feats'].shape}")
            print(f"  - 体素坐标形状: {batch['voxel_coords'].shape}")
            
            batch_size = len(batch['batch_offsets']) - 1
            
            # 在体素上聚合特征 (Mean Pooling)
            # 使用 C++ 扩展加速
            voxel_feats = pointgroup_ops.voxelization(batch['feats'], batch['v2p_map'])
            
            # 构建稀疏张量输入 (SparseConvTensor)
            input_tensor = spconv.SparseConvTensor(voxel_feats, batch['voxel_coords'].int(), batch['spatial_shape'], batch_size)
            
            # --- 步骤 2: Sparse U-Net 前向传播 ---
            print(f"\n[步骤 2] Sparse U-Net 前向传播:")
            # 输入卷积
            x = self.model.input_conv(input_tensor)
            # U-Net 主干
            x, _ = self.model.unet(x)
            # 输出层
            x = self.model.output_layer(x)
            
            print(f"  - 稀疏输出特征形状: {x.features.shape}")
            
            # --- 步骤 3: 映射回点 (Devoxelization) ---
            print(f"\n[步骤 3] 映射回点特征 (Devoxelization):")
            # 使用 p2v_map (point-to-voxel) 索引将体素特征映射回每个点
            p2v_map = batch['p2v_map'].long()
            
            # 检查无效索引 (-1 表示该点未被体素化，通常是极少数离群点)
            min_idx = p2v_map.min().item()
            
            if min_idx < 0:
                print("  - [Warning] 检测到未被体素化的点 (索引 -1)，填充零特征...")
                valid_mask = p2v_map >= 0
                point_features = torch.zeros((p2v_map.shape[0], x.features.shape[1]), device=self.device, dtype=x.features.dtype)
                if valid_mask.any():
                    point_features[valid_mask] = x.features[p2v_map[valid_mask]]
            else:
                point_features = x.features[p2v_map]
                
            print(f"  - 点级特征形状: {point_features.shape}")
            
            # --- 步骤 4: 超点池化 (Superpoint Pooling) ---
            print(f"\n[步骤 4] 超点池化 (Superpoint Pooling):")
            print(f"  - 超点索引形状: {batch['superpoints'].shape}")
            
            # 根据模型配置选择 Pooling 方式
            if self.model.pool == 'mean':
                sp_feats = scatter_mean(point_features, batch['superpoints'], dim=0)
            elif self.model.pool == 'max':
                sp_feats, _ = scatter_max(point_features, batch['superpoints'], dim=0)
                
            print(f"  - 最终超点特征形状: {sp_feats.shape}")
            
            return point_features, sp_feats

def get_args():
    # 获取脚本所在目录，用于构建默认配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'configs', 'spf_scannet.yaml')

    parser = argparse.ArgumentParser('SPFormer 特征提取器')
    parser.add_argument('--config', type=str, default=default_config, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='权重文件路径')
    parser.add_argument('--input_file', type=str, required=True, help='输入点云文件路径 (txt)')
    parser.add_argument('--output_file', type=str, default='extracted_features.pth', help='输出特征文件路径 (.pth)')
    parser.add_argument('--n_superpoints', type=int, default=512, help='生成的超点数量')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # 1. 初始化提取器
    try:
        extractor = FeatureExtractor(args.config, args.checkpoint)
    except Exception as e:
        print(f"[Error] 初始化失败: {e}")
        exit(1)
    
    # 2. 加载数据
    if not os.path.exists(args.input_file):
        print(f"[Error] 输入文件不存在: {args.input_file}")
        exit(1)
        
    xyz, rgb = extractor.load_data(args.input_file)
    
    # 3. 生成超点
    superpoints = extractor.generate_superpoints(xyz, rgb, n_clusters=args.n_superpoints)
    
    # 4. 提取特征
    point_feats, sp_feats = extractor.extract(xyz, rgb, superpoints)
    
    # 5. 保存结果
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save({
        'xyz': xyz, # 保存原始坐标以便可视化
        'rgb': rgb, # 保存原始颜色
        'superpoints': superpoints,
        'point_features': point_feats.cpu(),
        'superpoint_features': sp_feats.cpu()
    }, args.output_file)
    
    print(f"\n[Success] 特征已保存至: {args.output_file}")
