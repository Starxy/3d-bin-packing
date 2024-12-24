from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from algorithms.base import BinPackingAlgorithm, Item, Bin, ItemTooLargeError

class RandomPlacement(BinPackingAlgorithm):
    """随机放置算法"""
    
    def __init__(self, container_size: Tuple[int, int, int], seed: Optional[int] = None):
        super().__init__(container_size, seed)
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
    def find_random_position(self, bin: Bin, item: Item) -> Tuple[int, int, int]:
        """
        在容器中随机寻找可放置位置
        
        Args:
            bin: 目标容器
            item: 待放置物品
            
        Returns:
            随机可放置位置 (x, y, z)，如果找不到则返回 None
        """
        # 计算可能的位置范围
        max_x = bin.length - item.length + 1
        max_y = bin.width - item.width + 1
        max_z = bin.height - item.height + 1
        
        if max_x <= 0 or max_y <= 0 or max_z <= 0:
            return None
            
        # 随机尝试50次
        for _ in range(50):
            x = self.rng.randint(0, max_x)
            y = self.rng.randint(0, max_y)
            z = self.rng.randint(0, max_z)
            
            if bin.can_place_item(item, (x, y, z)):
                return (x, y, z)
                
        return None
        
    def pack(self) -> Dict[str, Any]:
        """
        执行随机放置算法
        
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - num_bins: int, 使用的容器数量
                - total_items: int, 物品总数
                - overall_utilization: float, 总体空间利用率
                - bins: List[Dict], 每个容器的详细信息
        """
        # 随机打乱物品顺序
        items = self.items.copy()
        self.rng.shuffle(items)
        
        current_bin = self.create_new_bin()
        
        for item in items:
            placed = False
            
            # 先尝试在现有容器中放置
            for bin in self.bins:
                position = self.find_random_position(bin, item)
                if position:
                    bin.place_item(item, position)
                    placed = True
                    break
            
            # 如果现有容器都放不下，创建新容器
            if not placed:
                new_bin = self.create_new_bin()
                position = self.find_random_position(new_bin, item)
                
                # 如果新容器也放不下，说明物品尺寸超过容器
                if not position:
                    raise ItemTooLargeError(
                        f"Item {item.id} is too large to fit in any container"
                    )
                    
                new_bin.place_item(item, position)
                
        return self.get_result()

def main():
    """测试随机放置算法"""
    from .test_utils import test_algorithm
    test_algorithm(RandomPlacement) 