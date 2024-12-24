from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
from algorithms.base import BinPackingAlgorithm, Item, Bin, ItemTooLargeError

class FirstFit(BinPackingAlgorithm):
    """优化的First Fit算法"""
    
    def __init__(self, container_size: Tuple[int, int, int], seed: Optional[int] = None):
        super().__init__(container_size, seed)
        self.position_cache = {}  # 添加位置缓存
        
    def _get_candidate_positions(self, bin: Bin, item: Item) -> List[Tuple[int, int, int]]:
        """获取候选放置位置"""
        positions = set()
        
        # 添加(0,0,0)作为初始位置
        positions.add((0, 0, 0))
        
        # 遍历已放置的物品
        for placed_item in bin.items:
            x, y, z = placed_item.position
            
            # 添加物品顶部位置
            positions.add((x, y, z + placed_item.height))
            
            # 添加物品侧面位置
            positions.add((x + placed_item.length, y, z))
            positions.add((x, y + placed_item.width, z))
        
        # 过滤掉超出容器范围的位置
        valid_positions = []
        for x, y, z in positions:
            if (x + item.length <= bin.length and 
                y + item.width <= bin.width and 
                z + item.height <= bin.height):
                valid_positions.append((x, y, z))
        
        return valid_positions
        
    def find_position(self, bin: Bin, item: Item) -> Optional[Tuple[int, int, int]]:
        """优化的位置查找算法"""
        # 使用缓存
        cache_key = (item.id, bin.id)
        if cache_key in self.position_cache:
            return self.position_cache[cache_key]
            
        # 快速检查：如果物品体积大于剩余空间，直接返回None
        remaining_volume = bin.volume * (1 - bin.space_utilization)
        if item.volume > remaining_volume:
            self.position_cache[cache_key] = None
            return None
            
        # 获取候选位置
        candidate_positions = self._get_candidate_positions(bin, item)
        
        # 优先考虑底部位置
        for x, y, z in candidate_positions:
            if z == 0 and bin.can_place_item(item, (x, y, z)):
                self.position_cache[cache_key] = (x, y, z)
                return (x, y, z)
        
        # 然后考虑其他位置
        for x, y, z in candidate_positions:
            if bin.can_place_item(item, (x, y, z)):
                self.position_cache[cache_key] = (x, y, z)
                return (x, y, z)
        
        self.position_cache[cache_key] = None
        return None
        
    def pack(self) -> Dict:
        """执行优化的First Fit算法"""
        # 预处理：检查物品是否都能放入容器
        print("检查物品尺寸...")
        for item in self.items:
            if (item.length > self.container_size[0] or
                item.width > self.container_size[1] or
                item.height > self.container_size[2]):
                raise ItemTooLargeError(
                    f"Item {item.id} is too large for the container"
                )
        
        print("对物品进行排序...")
        # 按体积降序排列物品，优先放置大物品
        sorted_items = sorted(
            self.items,
            key=lambda x: (-x.volume, -x.height, -x.width, -x.length)  # 多级排序
        )
        
        # 创建第一个容器
        self.create_new_bin()
        
        # 使用分组策略：相似大小的物品放在一起
        current_volume_group = sorted_items[0].volume if sorted_items else 0
        current_bin = self.bins[0]
        
        # 创建进度条
        pbar = tqdm(sorted_items, desc="放置物品", unit="个")
        try:
            for item in pbar:
                # 如果物品体积差异大，考虑使用新容器
                if current_volume_group / item.volume > 4:  # 体积比阈值
                    current_volume_group = item.volume
                    current_bin = self.create_new_bin()
                
                placed = False
                
                # 首先尝试当前容器
                position = self.find_position(current_bin, item)
                if position:
                    current_bin.place_item(item, position)
                    placed = True
                else:
                    # 尝试其他现有容器
                    for bin in reversed(self.bins):  # 从后往前查找，优先填满新容器
                        if bin == current_bin:
                            continue
                        position = self.find_position(bin, item)
                        if position:
                            bin.place_item(item, position)
                            placed = True
                            break
                
                # 如果现有容器都放不下，创建新容器
                if not placed:
                    new_bin = self.create_new_bin()
                    position = self.find_position(new_bin, item)
                    
                    if not position:
                        raise ItemTooLargeError(
                            f"Item {item.id} cannot be placed in any container"
                        )
                        
                    new_bin.place_item(item, position)
                    current_bin = new_bin
                    current_volume_group = item.volume
                
                # 更新进度条信息
                pbar.set_postfix({
                    'bins': len(self.bins),
                    'utilization': f"{self.bins[-1].space_utilization:.2%}"
                })
                
        finally:
            pbar.close()
        
        return self.get_result()
