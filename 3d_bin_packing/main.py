from pathlib import Path
import json
import sys

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.data_generator import DataGenerator
from evaluation.evaluator import Evaluator

def run_evaluation(size_name: str = 'small'):
    """运行算法评估
    
    Args:
        size_name: 数据集规模（small/medium/large）
    """
    print(f"\n开始评估{size_name}规模数据集...")
    
    # 创建评估器
    dataset_path = str(Path(__file__).parent / f'data/datasets/{size_name}.json')
    evaluator = Evaluator(
        dataset_path=dataset_path,
        num_runs=30,
        save_dir=f'results/{size_name}'
    )
    
    # 运行评估
    results = evaluator.run_evaluation()
    
    print(f"评估完成！")
    return results

def analyze_results():
    """分析所有结果"""
    print("\n开始分析结果...")
    
    # 加载所有结果
    results_dir = Path(__file__).parent / 'results'
    all_results = {}
    
    for size_name in ['small', 'medium', 'large']:
        result_file = results_dir / size_name / 'evaluation_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[size_name] = json.load(f)
    
    # 打印汇总结果
    for size_name, results in all_results.items():
        print(f"\n{size_name.capitalize()}规模数据集结果汇总:")
        for algo_name, algo_results in results.items():
            print(f"\n{algo_name}:")
            print(f"平均容器数量: {algo_results['num_bins']['mean']:.2f} ± {algo_results['num_bins']['std']:.2f}")
            print(f"平均空间利用率: {algo_results['utilization']['mean']:.2%} ± {algo_results['utilization']['std']:.2%}")
            print(f"平均运行时间: {algo_results['time']['mean']:.3f}秒")

def main():
    """主函数"""
    # 1. 生成数据集
    print("正在生成数据集...")
    from data.data_generator import DataGenerator
    generator = DataGenerator()
    generator.main()
    
    # 2. 运行评估
    sizes = ['small', 'medium', 'large']
    
    all_results = {}
    for size_name in sizes:
        all_results[size_name] = run_evaluation(size_name)
    
    # 3. 分析结果
    analyze_results()

if __name__ == '__main__':
    main() 