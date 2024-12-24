import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

class DataGenerator:
    """数据集生成器类"""
    
    def __init__(self, 
                 container_size: Tuple[int, int, int] = (100, 100, 100),
                 seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            container_size: 容器的尺寸 (长, 宽, 高)
            seed: 随机数种子
        """
        self.container_size = container_size
        self.seed = seed
        np.random.seed(seed)
        
    def _generate_items(self, 
                      num_items: int,
                      min_size: int = 10,
                      max_size: int = 50) -> List[Dict]:
        """
        生成指定数量的物品
        
        Args:
            num_items: 物品数量
            min_size: 物品最小尺寸
            max_size: 物品最大尺寸
            
        Returns:
            包含物品信息的列表，每个物品为字典格式
        """
        items = []
        for i in range(num_items):
            length = np.random.randint(min_size, max_size)
            width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            
            # 确保物品尺寸不超过容器
            length = min(length, self.container_size[0])
            width = min(width, self.container_size[1])
            height = min(height, self.container_size[2])
            
            items.append({
                'id': i,
                'length': length,
                'width': width,
                'height': height,
                'volume': length * width * height
            })
        
        return items
    
    def _generate_single_dataset(self, size_name: str) -> Dict:
        """
        生成单个数据集的私有方法
        
        Args:
            size_name: 数据集规模名称（small/medium/large）
            
        Returns:
            包含数据集信息的字典
        """
        # 根据规模确定物品数量范围
        size_ranges = {
            'small': (20, 50),
            'medium': (100, 200),
            'large': (500, 800)
        }
        
        min_items, max_items = size_ranges[size_name]
        num_items = np.random.randint(min_items, max_items + 1)
        
        dataset = {
            'container': {
                'length': self.container_size[0],
                'width': self.container_size[1],
                'height': self.container_size[2]
            },
            'size_name': size_name,
            'num_items': num_items,
            'items': self._generate_items(num_items)
        }
        return dataset
    
    def _save_dataset(self, dataset: Dict, size_name: str) -> str:
        """
        保存数据集到JSON文件的私有方法
        
        Args:
            dataset: 数据集字典
            size_name: 数据集规模名称
            
        Returns:
            保存的文件名
        """
        timestamp = int(time.time())
        filename = f'{timestamp}.json'
        
        save_path = Path(__file__).parent / 'datasets' / size_name / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        return filename
    
    def generate_datasets(self, size_name: str, num_instances: int = 3) -> List[str]:
        """
        生成并保存指定规模的多个数据集实例
        
        Args:
            size_name: 数据集规模名称（small/medium/large）
            num_instances: 需要生成的实例数量
            
        Returns:
            生成的数据集文件名列表
        """
        if size_name not in ['small', 'medium', 'large']:
            raise ValueError("size_name must be one of 'small', 'medium', 'large'")
            
        generated_files = []
        for i in range(num_instances):
            dataset = self._generate_single_dataset(size_name)
            filename = self._save_dataset(dataset, size_name)
            generated_files.append(filename)
            
        return generated_files
    
    @staticmethod
    def load_dataset(size_name: str, filename: str) -> Dict:
        """
        从JSON文件加载数据集
        
        Args:
            size_name: 数据集规模名称（small/medium/large）
            filename: 数据集文件名
            
        Returns:
            数据集字典
        """
        load_path = Path(__file__).parent / 'datasets' / size_name / filename
        
        with open(load_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset

def main():
    """测试数据生成器"""
    generator = DataGenerator()
    
    # 生成所有规模的数据集
    for size_name in ['small', 'medium', 'large']:
        print(f"\n生成{size_name}规模数据集:")
        files = generator.generate_datasets(size_name, num_instances=3)
        for i, filename in enumerate(files, 1):
            print(f"  生成第 {i} 个实例: {filename}")
    
    print("\n数据集生成完成！")
    
    # 验证数据集
    test_dir = Path(__file__).parent / 'datasets' / 'small'
    test_file = next(test_dir.glob('*.json'))
    loaded_dataset = DataGenerator.load_dataset('small', test_file.name)
    print(f"\n验证数据集:")
    print(f"容器尺寸: {loaded_dataset['container']}")
    print(f"数据集规模: {loaded_dataset['size_name']}")
    print(f"物品数量: {loaded_dataset['num_items']}")
    print(f"实际物品数量: {len(loaded_dataset['items'])}")

if __name__ == '__main__':
    main() 