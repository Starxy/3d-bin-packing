"""算法测试工具"""

# 添加项目根目录到Python路径
import sys
import os
from pathlib import Path

# 获取项目根目录（3d_bin_packing 文件夹）
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
    
import json
from typing import Type, Optional
from algorithms.base import BinPackingAlgorithm
from data.data_generator import DataGenerator

def load_test_dataset(size_name: str) -> dict:
    """
    加载指定规模的测试数据集
    
    Args:
        size_name: 数据集规模名称 ('small', 'medium', 'large')
        
    Returns:
        数据集字典
    """
    # 获取指定规模文件夹下的第一个数据集
    dataset_dir = Path(project_root) / 'data/datasets' / size_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"找不到{size_name}规模的数据集目录")
    
    # 获取文件夹中的第一个json文件
    json_files = list(dataset_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"在{size_name}目录下找不到数据集文件")
    
    # 加载数据集
    return DataGenerator.load_dataset(size_name, json_files[0].name)

def test_algorithm(algorithm_class: Type[BinPackingAlgorithm], 
                  dataset_size: str = 'small',
                  dataset: Optional[dict] = None) -> dict:
    """
    测试指定的算法
    
    Args:
        algorithm_class: 要测试的算法类
        dataset_size: 数据集大小 ('small', 'medium', 'large')
        dataset: 可选的数据集字典，如果提供则直接使用而不加载文件
        
    Returns:
        算法执行结果字典
    """
    # 获取数据集
    if dataset is None:
        dataset = load_test_dataset(dataset_size)
    
    # 创建算法实例
    container_size = (
        dataset['container']['length'],
        dataset['container']['width'],
        dataset['container']['height']
    )
    algorithm = algorithm_class(container_size, seed=42)
    
    # 加载数据并执行算法
    algorithm.load_items(dataset['items'])
    result = algorithm.pack()
    # 打印结果
    # 创建输出文件
    output_file = Path(f'{algorithm_class.__name__}_results.txt')
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"测试 {algorithm_class.__name__}:\n")
        f.write(f"数据集规模: {dataset['size_name']}\n")
        f.write(f"物品数量: {len(dataset['items'])}\n")
        f.write(f"使用容器数量: {result['num_bins']}\n")
        f.write(f"总体空间利用率: {result['overall_utilization']:.2%}\n")
        
        # 写入每个容器的详细信息
        f.write("\n各容器详细信息:\n")
        for bin_info in result['bins']:
            f.write(f"容器 {bin_info['id']}:\n")
            f.write(f"  - 空间利用率: {bin_info['utilization']:.2%}\n")
            f.write(f"  - 已放置物品数量: {len(bin_info['items'])}个\n")
    return result

def main():
    """测试算法"""
    
    # 测试所有规模的数据集
    from algorithms.random_placement import RandomPlacement
    from algorithms.first_fit import FirstFit
    from algorithms.worst_fit import WorstFit
    from algorithms.genetic_algorithm import GeneticAlgorithm
    for size in ['small', 'medium', 'large']:
        print(f"\n{'='*20} 测试 RandomPlacement {size}规模数据集 {'='*20}")
        # test_algorithm(RandomPlacement, dataset_size=size)
        # test_algorithm(FirstFit, dataset_size=size)
        # test_algorithm(WorstFit, dataset_size=size)
        test_algorithm(GeneticAlgorithm, dataset_size=size)
    

if __name__ == '__main__':
    main()