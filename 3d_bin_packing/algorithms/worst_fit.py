from typing import Dict, List, Tuple, Optional
from algorithms.base import BinPackingAlgorithm, Item, Bin, ItemTooLargeError
from tqdm import tqdm

class WorstFit(BinPackingAlgorithm):
    """Worst Fit算法实现
    
    该算法总是尝试将物品放入剩余空间最大的容器中。
    如果现有容器都无法放置，则创建新容器。
    """
    
    def __init__(self, container_size: Tuple[int, int, int], seed: Optional[int] = None):
        super().__init__(container_size, seed)
        self.position_cache = {}  # 位置缓存
        
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
        """查找物品在容器中的放置位置"""
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

    def find_best_bin(self, item: Item) -> Optional[Tuple[Bin, Tuple[int, int, int]]]:
        """查找最适合放置物品的容器（剩余空间最大的容器）"""
        best_bin = None
        best_position = None
        max_remaining_volume = -1

        for bin in self.bins:
            remaining_volume = bin.volume * (1 - bin.space_utilization)
            position = self.find_position(bin, item)
            
            if position and remaining_volume > max_remaining_volume:
                max_remaining_volume = remaining_volume
                best_bin = bin
                best_position = position

        if best_bin:
            return (best_bin, best_position)
        return None
        
    def pack(self) -> Dict:
        """执行Worst Fit算法"""
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
        # 按体积降序排列物品
        sorted_items = sorted(
            self.items,
            key=lambda x: (-x.volume, -x.height, -x.width, -x.length)
        )
        
        # 创建第一个容器
        self.create_new_bin()
        
        # 创建进度条
        pbar = tqdm(sorted_items, desc="放置物品", unit="个")
        try:
            for item in pbar:
                # 查找最佳容器
                best_result = self.find_best_bin(item)
                
                if best_result:
                    best_bin, position = best_result
                    best_bin.place_item(item, position)
                else:
                    # 创建新容器
                    new_bin = self.create_new_bin()
                    position = self.find_position(new_bin, item)
                    
                    if not position:
                        raise ItemTooLargeError(
                            f"Item {item.id} cannot be placed in any container"
                        )
                    
                    new_bin.place_item(item, position)
                
                # 更新进度条信息
                pbar.set_postfix({
                    'bins': len(self.bins),
                    'utilization': f"{self.bins[-1].space_utilization:.2%}"
                })
                
        finally:
            pbar.close()
        
        return self.get_result() 