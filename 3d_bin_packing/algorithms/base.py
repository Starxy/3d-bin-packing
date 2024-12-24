from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

class Item:
    """物品类"""
    def __init__(self, id: int, length: int, width: int, height: int):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height
        self.position: Optional[Tuple[int, int, int]] = None  # (x, y, z)
        self.bin_id: Optional[int] = None  # 物品所在容器的ID

    def __repr__(self):
        return f"Item(id={self.id}, size=({self.length},{self.width},{self.height}), pos={self.position}, bin={self.bin_id})"

class Bin:
    """容器类"""
    def __init__(self, id: int, length: int, width: int, height: int):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height
        self.items: List[Item] = []  # 存储在该容器中的物品
        self.space_utilization = 0.0  # 空间利用率
        
        # 用于记录已使用空间的3D矩阵
        self.space_matrix = np.zeros((length, width, height), dtype=bool)
    
    def get_used_volume(self) -> float:
        """计算已使用体积"""
        return sum(item.volume for item in self.items)
    
    def update_space_utilization(self):
        """更新空间利用率"""
        self.space_utilization = self.get_used_volume() / self.volume
        
    def can_place_item(self, item: Item, position: Tuple[int, int, int]) -> bool:
        """检查在指定位置是否可以放置物品"""
        x, y, z = position
        
        # 检查是否超出容器边界
        if (x + item.length > self.length or 
            y + item.width > self.width or 
            z + item.height > self.height):
            return False
            
        # 检查是否与其他物品重叠
        space_needed = self.space_matrix[
            x:x+item.length,
            y:y+item.width,
            z:z+item.height
        ]
        return not np.any(space_needed)
    
    def place_item(self, item: Item, position: Tuple[int, int, int]) -> bool:
        """在指定位置放置物品"""
        if not self.can_place_item(item, position):
            return False
            
        x, y, z = position
        # 更新空间占用矩阵
        self.space_matrix[
            x:x+item.length,
            y:y+item.width,
            z:z+item.height
        ] = True
        
        # 更新物品信息
        item.position = position
        item.bin_id = self.id
        self.items.append(item)
        
        # 更新利用率
        self.update_space_utilization()
        return True

class BinPackingAlgorithm(ABC):
    """装箱算法基类"""
    
    def __init__(self, container_size: Tuple[int, int, int], seed: Optional[int] = None):
        """
        初始化算法
        
        Args:
            container_size: 容器的尺寸 (length, width, height)
            seed: 随机数种子，用于保证结果可重现
        """
        self.container_size = container_size
        self.bins: List[Bin] = []
        self.items: List[Item] = []
        if seed is not None:
            np.random.seed(seed)
    
    def load_items(self, items_data: List[Dict[str, int]]):
        """
        加载物品数据
        
        Args:
            items_data: 物品数据列表，每个物品必须包含 id, length, width, height 字段
            
        Raises:
            ValueError: 如果物品数据格式不正确
        """
        required_fields = {'id', 'length', 'width', 'height'}
        
        for item in items_data:
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(
                    f"物品数据缺少必要字段: {missing_fields}"
                )
            
            if not all(isinstance(item[field], int) for field in required_fields):
                raise ValueError(
                    f"物品的id和尺寸必须是整数"
                )
        
        self.items = [
            Item(
                id=item['id'],
                length=item['length'],
                width=item['width'],
                height=item['height']
            )
            for item in items_data
        ]
    
    def create_new_bin(self) -> Bin:
        """创建新容器"""
        bin_id = len(self.bins)
        new_bin = Bin(
            id=bin_id,
            length=self.container_size[0],
            width=self.container_size[1],
            height=self.container_size[2]
        )
        self.bins.append(new_bin)
        return new_bin
    
    @abstractmethod
    def pack(self) -> Dict[str, Any]:
        """
        执行装箱算法
        
        Returns:
            Dict[str, Any]: 包含以下键的字典：
                - num_bins: int, 使用的容器数量
                - total_items: int, 物品总数
                - overall_utilization: float, 总体空间利用率
                - bins: List[Dict], 每个容器的详细信息
        """
        pass
    
    def get_result(self) -> Dict:
        """获取装箱结果"""
        total_item_volume = sum(item.volume for item in self.items)
        total_bin_volume = sum(bin.volume for bin in self.bins)
        
        # 如果有容器，则生成第一个容器的3D示意图
        if self.bins:
            self._save_first_bin_visualization()
        
        return {
            'num_bins': len(self.bins),
            'total_items': len(self.items),
            'overall_utilization': total_item_volume / total_bin_volume if self.bins else 0,
            'bins': [
                {
                    'id': bin.id,
                    'utilization': bin.space_utilization,
                    'items': [
                        {
                            'id': item.id,
                            'position': item.position,
                            'size': (item.length, item.width, item.height)
                        }
                        for item in bin.items
                    ]
                }
                for bin in self.bins
            ]
        }
    
    def _save_first_bin_visualization(self):
        """生成并保存第一个容器的3D填充示意图"""
        if not self.bins:
            return
        
        bin_obj = self.bins[0]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制容器边框
        container_points = np.array([
            [0, 0, 0],
            [bin_obj.length, 0, 0],
            [bin_obj.length, bin_obj.width, 0],
            [0, bin_obj.width, 0],
            [0, 0, bin_obj.height],
            [bin_obj.length, 0, bin_obj.height],
            [bin_obj.length, bin_obj.width, bin_obj.height],
            [0, bin_obj.width, bin_obj.height]
        ])
        
        # 定义容器的面
        container_faces = [
            [container_points[j] for j in [0, 1, 2, 3]],  # 底面
            [container_points[j] for j in [4, 5, 6, 7]],  # 顶面
            [container_points[j] for j in [0, 1, 5, 4]],  # 前面
            [container_points[j] for j in [2, 3, 7, 6]],  # 后面
            [container_points[j] for j in [0, 3, 7, 4]],  # 左面
            [container_points[j] for j in [1, 2, 6, 5]]   # 右面
        ]
        
        # 绘制半透明的容器
        container = Poly3DCollection(container_faces, alpha=0.2)
        container.set_facecolor('gray')
        ax.add_collection3d(container)
        
        # 为每个物品生成随机颜色并绘制
        np.random.seed(42)  # 固定随机种子以保持颜色一致
        for item in bin_obj.items:
            if item.position is None:
                continue
            
            x, y, z = item.position
            color = np.random.rand(3,)
            
            # 计算物品的顶点
            item_points = np.array([
                [x, y, z],
                [x + item.length, y, z],
                [x + item.length, y + item.width, z],
                [x, y + item.width, z],
                [x, y, z + item.height],
                [x + item.length, y, z + item.height],
                [x + item.length, y + item.width, z + item.height],
                [x, y + item.width, z + item.height]
            ])
            
            # 定义物品的面
            item_faces = [
                [item_points[j] for j in [0, 1, 2, 3]],
                [item_points[j] for j in [4, 5, 6, 7]],
                [item_points[j] for j in [0, 1, 5, 4]],
                [item_points[j] for j in [2, 3, 7, 6]],
                [item_points[j] for j in [0, 3, 7, 4]],
                [item_points[j] for j in [1, 2, 6, 5]]
            ]
            
            # 绘制物品
            item_poly = Poly3DCollection(item_faces, alpha=0.6)
            item_poly.set_facecolor(color)
            ax.add_collection3d(item_poly)
        # 设置坐标轴
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        ax.set_xlabel('长度')
        ax.set_ylabel('宽度')
        ax.set_zlabel('高度')
        
        # 设置视角和显示范围
        ax.view_init(elev=30, azim=45)
        ax.set_xlim([0, bin_obj.length])
        ax.set_ylim([0, bin_obj.width])
        ax.set_zlim([0, bin_obj.height])
        
        plt.title(f'容器 #{bin_obj.id} 填充示意图 (利用率: {bin_obj.space_utilization:.2%})')
        # 保存图片,使用时间戳命名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'bin_{bin_obj.id}_visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

class BinPackingError(Exception):
    """装箱算法异常基类"""
    pass

class ItemTooLargeError(BinPackingError):
    """物品尺寸超过容器异常"""
    pass

class InvalidItemDataError(BinPackingError):
    """无效的物品数据异常"""
    pass