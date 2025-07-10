import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def load_data_in_chunks(file_path, chunk_size=1000):
    """分块读取大JSON文件，避免内存溢出"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)
    return data

def analyze_cascade_structure(edge_index):
    """分析级联网络结构特征"""
    if not edge_index or len(edge_index) != 2:
        return {
            'num_edges': 0,
            'num_nodes': 0,
            'max_depth': 0,
            'avg_degree': 0,
            'source_nodes': 0
        }
    
    # 边数量
    num_edges = len(edge_index[0])
    
    # 节点数量（所有出现过的节点）
    all_nodes = set(edge_index[0] + edge_index[1])
    num_nodes = len(all_nodes)
    
    # 统计每个节点的出度和入度
    out_degree = Counter(edge_index[0])
    in_degree = Counter(edge_index[1])
    
    # 源节点数量（只有出度没有入度的节点）
    source_nodes = len([node for node in all_nodes if node in out_degree and node not in in_degree])
    
    # 平均度数
    total_degree = sum(out_degree.values()) + sum(in_degree.values())
    avg_degree = total_degree / num_nodes if num_nodes > 0 else 0
    
    # 估算最大深度（简化计算）
    max_depth = estimate_max_depth(edge_index)
    
    return {
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'max_depth': max_depth,
        'avg_degree': avg_degree,
        'source_nodes': source_nodes
    }

def estimate_max_depth(edge_index):
    """估算网络的最大深度"""
    if not edge_index or len(edge_index[0]) == 0:
        return 0
    
    # 构建邻接表
    graph = defaultdict(list)
    for i in range(len(edge_index[0])):
        source = edge_index[0][i]
        target = edge_index[1][i]
        graph[source].append(target)
    
    # 找到源节点（入度为0的节点）
    all_nodes = set(edge_index[0] + edge_index[1])
    target_nodes = set(edge_index[1])
    source_nodes = all_nodes - target_nodes
    
    if not source_nodes:
        return 1
    
    # BFS计算最大深度
    max_depth = 0
    for source in source_nodes:
        depth = bfs_max_depth(graph, source)
        max_depth = max(max_depth, depth)
    
    return max_depth

def bfs_max_depth(graph, start_node):
    """BFS计算从起始节点的最大深度"""
    if start_node not in graph:
        return 1
    
    queue = [(start_node, 1)]
    visited = set([start_node])
    max_depth = 1
    
    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return max_depth

def analyze_features(x_data):
    """分析节点特征"""
    if not x_data:
        return {
            'num_features': 0,
            'feature_dim': 0,
            'avg_feature_norm': 0
        }
    
    num_features = len(x_data)
    
    # 获取特征维度
    first_feature = next(iter(x_data.values()))
    feature_dim = len(first_feature) if first_feature else 0
    
    # 计算平均特征范数
    feature_norms = []
    for node_features in x_data.values():
        if node_features:
            norm = np.linalg.norm(node_features)
            feature_norms.append(norm)
    
    avg_feature_norm = np.mean(feature_norms) if feature_norms else 0
    
    return {
        'num_features': num_features,
        'feature_dim': feature_dim,
        'avg_feature_norm': avg_feature_norm
    }

def main():
    print("开始加载和分析数据...")
    
    # 加载训练数据
    try:
        train_data = load_data_in_chunks('data/train.json')
        print(f"成功加载训练数据，共 {len(train_data)} 条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 初始化统计变量
    stats = {
        'total_samples': len(train_data),
        'label_distribution': Counter(),
        'edge_stats_by_label': defaultdict(list),
        'node_stats_by_label': defaultdict(list),
        'depth_stats_by_label': defaultdict(list),
        'feature_stats_by_label': defaultdict(list),
        'overall_edge_stats': [],
        'overall_node_stats': [],
        'overall_depth_stats': [],
        'feature_dim': 0
    }
    
    print("开始统计分析...")
    
    # 逐条分析数据
    for i, sample in enumerate(train_data):
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(train_data)} 条记录")
        
        # 获取标签
        label = sample.get('label', -1)
        stats['label_distribution'][label] += 1
        
        # 分析网络结构
        edge_index = sample.get('edge_index', [])
        structure_stats = analyze_cascade_structure(edge_index)
        
        # 按标签分类统计
        stats['edge_stats_by_label'][label].append(structure_stats['num_edges'])
        stats['node_stats_by_label'][label].append(structure_stats['num_nodes'])
        stats['depth_stats_by_label'][label].append(structure_stats['max_depth'])
        
        # 整体统计
        stats['overall_edge_stats'].append(structure_stats['num_edges'])
        stats['overall_node_stats'].append(structure_stats['num_nodes'])
        stats['overall_depth_stats'].append(structure_stats['max_depth'])
        
        # 分析特征
        x_data = sample.get('x', {})
        feature_stats = analyze_features(x_data)
        stats['feature_stats_by_label'][label].append(feature_stats['num_features'])
        
        # 记录特征维度
        if feature_stats['feature_dim'] > 0:
            stats['feature_dim'] = feature_stats['feature_dim']
    
    # 打印统计结果
    print("\n" + "="*60)
    print("数据集统计分析结果")
    print("="*60)
    
    # 基本信息
    print(f"\n1. 基本信息:")
    print(f"   总样本数: {stats['total_samples']}")
    print(f"   特征维度: {stats['feature_dim']}")
    
    # 标签分布
    print(f"\n2. 标签分布:")
    for label, count in sorted(stats['label_distribution'].items()):
        percentage = count / stats['total_samples'] * 100
        label_name = "突发事件" if label == 1 else "正常事件" if label == 0 else f"未知标签({label})"
        print(f"   {label_name} (label={label}): {count} 条 ({percentage:.2f}%)")
    
    # 边数量统计
    print(f"\n3. 网络边数量统计:")
    print(f"   整体边数量范围: {min(stats['overall_edge_stats'])} - {max(stats['overall_edge_stats'])}")
    print(f"   整体边数量均值: {np.mean(stats['overall_edge_stats']):.2f}")
    print(f"   整体边数量中位数: {np.median(stats['overall_edge_stats']):.2f}")
    
    for label in sorted(stats['edge_stats_by_label'].keys()):
        edge_data = stats['edge_stats_by_label'][label]
        label_name = "突发事件" if label == 1 else "正常事件" if label == 0 else f"标签{label}"
        print(f"   {label_name}:")
        print(f"     最大边数: {max(edge_data)}")
        print(f"     平均边数: {np.mean(edge_data):.2f}")
        print(f"     中位数边数: {np.median(edge_data):.2f}")
    
    # 节点数量统计
    print(f"\n4. 网络节点数量统计:")
    print(f"   整体节点数量范围: {min(stats['overall_node_stats'])} - {max(stats['overall_node_stats'])}")
    print(f"   整体节点数量均值: {np.mean(stats['overall_node_stats']):.2f}")
    
    for label in sorted(stats['node_stats_by_label'].keys()):
        node_data = stats['node_stats_by_label'][label]
        label_name = "突发事件" if label == 1 else "正常事件" if label == 0 else f"标签{label}"
        print(f"   {label_name}:")
        print(f"     最大节点数: {max(node_data)}")
        print(f"     平均节点数: {np.mean(node_data):.2f}")
    
    # 网络深度统计
    print(f"\n5. 网络深度统计:")
    print(f"   整体深度范围: {min(stats['overall_depth_stats'])} - {max(stats['overall_depth_stats'])}")
    print(f"   整体深度均值: {np.mean(stats['overall_depth_stats']):.2f}")
    
    for label in sorted(stats['depth_stats_by_label'].keys()):
        depth_data = stats['depth_stats_by_label'][label]
        label_name = "突发事件" if label == 1 else "正常事件" if label == 0 else f"标签{label}"
        print(f"   {label_name}:")
        print(f"     最大深度: {max(depth_data)}")
        print(f"     平均深度: {np.mean(depth_data):.2f}")
    
    # 生成可视化图表
    print(f"\n6. 生成可视化图表...")
    generate_visualizations(stats)
    
    print("\n分析完成！")

def generate_visualizations(stats):
    """生成可视化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('突发事件级联分类数据集分析', fontsize=16)
    
    # 1. 标签分布
    labels = list(stats['label_distribution'].keys())
    counts = list(stats['label_distribution'].values())
    label_names = ["正常事件" if l == 0 else "突发事件" if l == 1 else f"标签{l}" for l in labels]
    
    axes[0, 0].pie(counts, labels=label_names, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('标签分布')
    
    # 2. 边数量分布对比
    edge_data_by_label = []
    edge_labels = []
    for label in sorted(stats['edge_stats_by_label'].keys()):
        edge_data_by_label.append(stats['edge_stats_by_label'][label])
        edge_labels.append("正常事件" if label == 0 else "突发事件" if label == 1 else f"标签{label}")
    
    axes[0, 1].boxplot(edge_data_by_label, labels=edge_labels)
    axes[0, 1].set_title('各类别边数量分布')
    axes[0, 1].set_ylabel('边数量')
    
    # 3. 节点数量分布对比
    node_data_by_label = []
    for label in sorted(stats['node_stats_by_label'].keys()):
        node_data_by_label.append(stats['node_stats_by_label'][label])
    
    axes[0, 2].boxplot(node_data_by_label, labels=edge_labels)
    axes[0, 2].set_title('各类别节点数量分布')
    axes[0, 2].set_ylabel('节点数量')
    
    # 4. 整体边数量分布直方图
    axes[1, 0].hist(stats['overall_edge_stats'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('整体边数量分布')
    axes[1, 0].set_xlabel('边数量')
    axes[1, 0].set_ylabel('频次')
    
    # 5. 整体节点数量分布直方图
    axes[1, 1].hist(stats['overall_node_stats'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('整体节点数量分布')
    axes[1, 1].set_xlabel('节点数量')
    axes[1, 1].set_ylabel('频次')
    
    # 6. 网络深度分布对比
    depth_data_by_label = []
    for label in sorted(stats['depth_stats_by_label'].keys()):
        depth_data_by_label.append(stats['depth_stats_by_label'][label])
    
    axes[1, 2].boxplot(depth_data_by_label, labels=edge_labels)
    axes[1, 2].set_title('各类别网络深度分布')
    axes[1, 2].set_ylabel('网络深度')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   可视化图表已保存为 'dataset_analysis.png'")

def check_first_edge_source(file_path):
    """统计edge_index[0][0]是否都等于0（在存在交互关系的前提下）"""
    print("开始检查edge_index[0][0]的分布...")
    
    try:
        train_data = load_data_in_chunks(file_path)
        print(f"成功加载数据，共 {len(train_data)} 条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    stats = {
        'total_samples': len(train_data),
        'samples_with_edges': 0,
        'first_edge_source_0': 0,
        'first_edge_source_not_0': 0,
        'first_edge_source_values': Counter(),
        'empty_edge_samples': 0
    }
    
    for i, sample in enumerate(train_data):
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(train_data)} 条记录")
        
        edge_index = sample.get('edge_index', [])
        
        # 检查是否有边
        if not edge_index or len(edge_index) != 2 or len(edge_index[0]) == 0:
            stats['empty_edge_samples'] += 1
            continue
        
        stats['samples_with_edges'] += 1
        
        # 获取第一条边的源节点
        first_source = edge_index[0][0]
        stats['first_edge_source_values'][first_source] += 1
        
        if first_source == 0:
            stats['first_edge_source_0'] += 1
        else:
            stats['first_edge_source_not_0'] += 1
    
    # 打印结果
    print("\n" + "="*60)
    print("Edge_index[0][0] 统计分析结果")
    print("="*60)
    
    print(f"\n总样本数: {stats['total_samples']}")
    print(f"空边样本数: {stats['empty_edge_samples']}")
    print(f"有边样本数: {stats['samples_with_edges']}")
    
    if stats['samples_with_edges'] > 0:
        print(f"\n在有交互关系的样本中:")
        print(f"edge_index[0][0] == 0 的样本数: {stats['first_edge_source_0']}")
        print(f"edge_index[0][0] != 0 的样本数: {stats['first_edge_source_not_0']}")
        print(f"edge_index[0][0] == 0 的比例: {stats['first_edge_source_0']/stats['samples_with_edges']*100:.2f}%")
        
        print(f"\nedge_index[0][0] 的值分布:")
        for value, count in sorted(stats['first_edge_source_values'].items()):
            percentage = count / stats['samples_with_edges'] * 100
            print(f"  值 {value}: {count} 次 ({percentage:.2f}%)")
        
        # 判断结论
        if stats['first_edge_source_0'] == stats['samples_with_edges']:
            print(f"\n✅ 结论: 所有有交互关系的样本中，edge_index[0][0] 都等于 0")
        else:
            print(f"\n❌ 结论: 不是所有有交互关系的样本中，edge_index[0][0] 都等于 0")
            print(f"   有 {stats['first_edge_source_not_0']} 个样本的 edge_index[0][0] 不等于 0")
    else:
        print("没有找到有交互关系的样本")

if __name__ == "__main__":
    main()  # 注释掉原来的main函数
    # check_first_edge_source('data/train.json')  # 执行新的判定函数