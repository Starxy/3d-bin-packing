from typing import Dict, List, Type
import json
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.base import BinPackingAlgorithm
from algorithms.random_placement import RandomPlacement
from algorithms.first_fit import FirstFit
from algorithms.next_fit import NextFit
from algorithms.genetic_algorithm import GeneticAlgorithm

class Evaluator:
    """算法评估器"""
    
    def __init__(self, 
                 dataset_path: str,
                 num_runs: int = 30,
                 save_dir: str = 'results'):
        """
        初始化评估器
        
        Args:
            dataset_path: 数据集路径
            num_runs: 每个算法运行次数
            save_dir: 结果保存目录
        """
        self.num_runs = num_runs
        self.save_dir = Path(__file__).parents[1] / save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
            
        # 获取容器尺寸
        self.container_size = (
            self.dataset['container']['length'],
            self.dataset['container']['width'],
            self.dataset['container']['height']
        )
        
        # 定义要评估的算法
        self.algorithms = {
            'Random': RandomPlacement,
            'FirstFit': FirstFit,
            'NextFit': NextFit,
            'GeneticAlgorithm': GeneticAlgorithm
        }
        
    def evaluate_algorithm(self, 
                         algorithm_class: Type[BinPackingAlgorithm],
                         items_data: List[Dict]) -> Dict:
        """
        评估单个算法在指定数据集上的表现
        
        Args:
            algorithm_class: 算法类
            items_data: 物品数据
            
        Returns:
            包含评估结果的字典
        """
        num_bins_list = []
        utilization_list = []
        time_list = []
        
        for run in range(self.num_runs):
            # 创建算法实例
            algorithm = algorithm_class(
                container_size=self.container_size,
                seed=run if hasattr(algorithm_class, 'seed') else None
            )
            
            # 加载数据并执行算法
            algorithm.load_items(items_data)
            
            start_time = time.time()
            result = algorithm.pack()
            end_time = time.time()
            
            # 记录结果
            num_bins_list.append(result['num_bins'])
            utilization_list.append(result['overall_utilization'])
            time_list.append(end_time - start_time)
            
        return {
            'num_bins': {
                'mean': np.mean(num_bins_list),
                'std': np.std(num_bins_list),
                'min': np.min(num_bins_list),
                'max': np.max(num_bins_list),
                'values': num_bins_list
            },
            'utilization': {
                'mean': np.mean(utilization_list),
                'std': np.std(utilization_list),
                'min': np.min(utilization_list),
                'max': np.max(utilization_list),
                'values': utilization_list
            },
            'time': {
                'mean': np.mean(time_list),
                'std': np.std(time_list),
                'total': sum(time_list)
            }
        }
        
    def run_evaluation(self):
        """执行完整的评估流程"""
        results = {}
        
        # 对每种规模的数据集评估
        for size in ['small', 'medium', 'large']:
            results[size] = {}
            items_data = self.dataset[size]
            
            # 评估每个算法
            for algo_name, algo_class in self.algorithms.items():
                print(f"评估 {algo_name} 在 {size} 数据集上的表现...")
                results[size][algo_name] = self.evaluate_algorithm(
                    algo_class, items_data
                )
                
        # 保存结果
        self.save_results(results)
        
        # 生成可视化
        self.generate_plots(results)
        
        return results
    
    def save_results(self, results: Dict):
        """保存评估结果到JSON文件"""
        save_path = self.save_dir / 'evaluation_results.json'
        
        # 转换numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj
            
        results = convert_to_native(results)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    def generate_plots(self, results: Dict):
        """生成评估结果的可视化图表"""
        # 设置图表风格
        plt.style.use('seaborn')
        
        # 1. 箱线图：比较不同算法的容器数量
        self._plot_boxplot(results, 'num_bins', '容器数量对比')
        
        # 2. 箱线图：比较不同算法的空间利用率
        self._plot_boxplot(results, 'utilization', '空间利用率对比')
        
        # 3. 柱状图：比较不同算法的运行时间
        self._plot_time_comparison(results)
        
    def _plot_boxplot(self, results: Dict, metric: str, title: str):
        """生成箱线图"""
        plt.figure(figsize=(12, 6))
        
        data = []
        labels = []
        
        for size in ['small', 'medium', 'large']:
            for algo in self.algorithms.keys():
                values = results[size][algo][metric]['values']
                data.append(values)
                labels.extend([f"{algo}\n({size})"] * len(values))
                
        plt.boxplot(data, labels=labels)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(self.save_dir / f"{metric}_boxplot.png")
        plt.close()
        
    def _plot_time_comparison(self, results: Dict):
        """生成运行时间对比图"""
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(self.algorithms))
        width = 0.25
        
        for i, size in enumerate(['small', 'medium', 'large']):
            times = [results[size][algo]['time']['mean'] 
                    for algo in self.algorithms.keys()]
            plt.bar(x + i*width, times, width, 
                   label=size)
            
        plt.xlabel('算法')
        plt.ylabel('平均运行时间 (秒)')
        plt.title('算法运行时间对比')
        plt.xticks(x + width, list(self.algorithms.keys()))
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(self.save_dir / "time_comparison.png")
        plt.close()

def main():
    """运行评估"""
    # 设置数据集路径
    dataset_path = Path(__file__).parents[1] / 'data/datasets/test_dataset.json'
    
    # 创建评估器
    evaluator = Evaluator(
        dataset_path=str(dataset_path),
        num_runs=30
    )
    
    # 运行评估
    results = evaluator.run_evaluation()
    
    # 打印部分结果
    for size in ['small', 'medium', 'large']:
        print(f"\n{size.capitalize()} 数据集结果:")
        for algo_name in evaluator.algorithms:
            result = results[size][algo_name]
            print(f"\n{algo_name}:")
            print(f"平均容器数量: {result['num_bins']['mean']:.2f} ± {result['num_bins']['std']:.2f}")
            print(f"平均空间利用率: {result['utilization']['mean']:.2%} ± {result['utilization']['std']:.2%}")
            print(f"平均运行时间: {result['time']['mean']:.3f}秒")

if __name__ == '__main__':
    main() 