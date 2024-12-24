from typing import Dict, List, Tuple, Optional
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from algorithms.base import BinPackingAlgorithm, Item, Bin, ItemTooLargeError

class Individual:
    """个体类，表示一个装箱方案"""
    def __init__(self, 
                 packing_sequence: List[int],
                 placement_strategy: Dict[str, float]):
        self.packing_sequence = packing_sequence  # 装箱顺序
        self.placement_strategy = placement_strategy  # 放置策略参数
        self.fitness = 0.0  # 适应度
        self.num_bins = 0  # 使用的容器数量
        self.bins: List[Bin] = []  # 装箱结果
        
class GeneticAlgorithm(BinPackingAlgorithm):
    """改进的遗传算法"""
    
    def __init__(self, 
                 container_size: Tuple[int, int, int],
                 population_size: int = 50,
                 generations: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2,
                 early_stop_generations: int = 10,  # 早停参数
                 seed: Optional[int] = None):
        super().__init__(container_size, seed)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.early_stop_generations = early_stop_generations
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
        # 添加位置缓存
        self.position_cache = {}
        
        # 初始化策略参数范围
        self.strategy_bounds = {
            'height_priority': (0.0, 1.0),
            'contact_area_priority': (0.0, 1.0),
            'volume_priority': (0.0, 1.0),
        }
        
    def initialize_population(self) -> List[Individual]:
        """使用启发式算法优化初始种群"""
        population = []
        
        # 使用First Fit算法生成一个较好的初始解
        from algorithms.first_fit import FirstFit
        ff = FirstFit(self.container_size, seed=self.rng.randint(0, 1000))
        ff.items = self.items
        ff.pack()
        
        # 将First Fit的解转换为个体
        ff_sequence = [item.id for bin in ff.bins for item in bin.items]
        ff_strategy = {
            'height_priority': 0.5,
            'contact_area_priority': 0.3,
            'volume_priority': 0.2
        }
        population.append(Individual(ff_sequence, ff_strategy))
        
        # 生成其余随机个体
        for _ in range(self.population_size - 1):
            packing_sequence = list(range(len(self.items)))
            self.rng.shuffle(packing_sequence)
            
            placement_strategy = {
                param: self.rng.uniform(low, high)
                for param, (low, high) in self.strategy_bounds.items()
            }
            
            population.append(Individual(packing_sequence, placement_strategy))
            
        return population
        
    def find_position(self, 
                     bin: Bin, 
                     item: Item, 
                     strategy: Dict[str, float]) -> Optional[Tuple[int, int, int]]:
        """优化的位置查找算法"""
        # 使用缓存
        cache_key = (item.id, bin.id, tuple(sorted(strategy.items())))
        if cache_key in self.position_cache:
            return self.position_cache[cache_key]

        best_position = None
        best_score = float('-inf')
        
        # 优化：只检查已放置物品的顶部和侧面
        candidate_positions = self._get_candidate_positions(bin, item)
        
        for x, y, z in candidate_positions:
            if not bin.can_place_item(item, (x, y, z)):
                continue
                
            score = self.evaluate_position(bin, item, (x, y, z), strategy)
            
            if score > best_score:
                best_score = score
                best_position = (x, y, z)
        
        # 缓存结果
        self.position_cache[cache_key] = best_position
        return best_position
        
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
        
    def evaluate_position(self, 
                         bin: Bin, 
                         item: Item, 
                         position: Tuple[int, int, int],
                         strategy: Dict[str, float]) -> float:
        """评估放置位置的得分"""
        x, y, z = position
        
        # 计算接触面积
        bottom_contact = 0
        if z == 0:  # 与容器底部接触
            bottom_contact = item.length * item.width
        else:  # 与其他物品接触
            for dx in range(item.length):
                for dy in range(item.width):
                    if bin.space_matrix[x+dx, y+dy, z-1]:
                        bottom_contact += 1
                        
        # 计算高度得分
        height_score = 1.0 - (z + item.height) / bin.height
        
        # 计算体积利用率得分
        volume_score = item.volume / (bin.length * bin.width * bin.height)
        
        # 综合评分
        return (strategy['height_priority'] * height_score +
                strategy['contact_area_priority'] * (bottom_contact / (item.length * item.width)) +
                strategy['volume_priority'] * volume_score)
        
    def decode_and_evaluate(self, individual: Individual) -> float:
        """解码并评估个体"""
        # 清空现有容器
        self.bins.clear()
        current_bin = self.create_new_bin()
        
        # 按照装箱顺序放置物品
        for item_idx in individual.packing_sequence:
            item = self.items[item_idx]
            placed = False
            
            # 尝试在现有容器中放置
            for bin in self.bins:
                position = self.find_position(bin, item, individual.placement_strategy)
                if position:
                    bin.place_item(item, position)
                    placed = True
                    break
                    
            # 如果现有容器都放不下，创建新容器
            if not placed:
                new_bin = self.create_new_bin()
                position = self.find_position(new_bin, item, individual.placement_strategy)
                if not position:
                    raise ItemTooLargeError(
                        f"Item {item.id} is too large to fit in any container"
                    )
                new_bin.place_item(item, position)
                
        # 记录结果
        individual.bins = deepcopy(self.bins)
        individual.num_bins = len(self.bins)
        
        # 计算适应度
        total_volume = sum(item.volume for item in self.items)
        volume_utilization = total_volume / (individual.num_bins * self.bins[0].volume)
        
        # 适应度 = 1/容器数量 + 空间利用率
        individual.fitness = (1.0 / individual.num_bins) + volume_utilization
        return individual.fitness
        
    def order_crossover(self, 
                       parent1: List[int], 
                       parent2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉"""
        size = len(parent1)
        # 随机选择交叉点
        point1, point2 = sorted(self.rng.choice(size, size=2, replace=False))
        
        def create_child(p1: List[int], p2: List[int]) -> List[int]:
            # 复制交叉区段
            child = [-1] * size
            for i in range(point1, point2):
                child[i] = p1[i]
                
            # 填充剩余位置
            current = point2
            for i in range(size):
                if p2[i] not in child:
                    if current >= size:
                        current = 0
                    while child[current] != -1:
                        current += 1
                        if current >= size:
                            current = 0
                    child[current] = p2[i]
                    current += 1
                    
            return child
            
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        return child1, child2
        
    def arithmetic_crossover(self, 
                           parent1: Dict[str, float], 
                           parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """算术交叉"""
        alpha = self.rng.uniform(0, 1)
        child1 = {}
        child2 = {}
        
        for key in parent1:
            child1[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
            child2[key] = (1 - alpha) * parent1[key] + alpha * parent2[key]
            
        return child1, child2
        
    def mutate(self, individual: Individual, temperature: float):
        """模拟退火突变"""
        # 序列突变
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.choice(len(individual.packing_sequence), size=2, replace=False)
            individual.packing_sequence[i], individual.packing_sequence[j] = \
                individual.packing_sequence[j], individual.packing_sequence[i]
                
        # 策略参数突变
        for param in individual.placement_strategy:
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.normal(0, temperature)  # 使用温度控制突变幅度
                low, high = self.strategy_bounds[param]
                individual.placement_strategy[param] = np.clip(
                    individual.placement_strategy[param] + delta,
                    low, high
                )
                
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """锦标赛选择"""
        tournament_size = 3
        selected = []
        
        while len(selected) < self.population_size - self.elite_size:
            tournament = self.rng.choice(population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(deepcopy(winner))
            
        return selected
        
    def pack(self) -> Dict:
        """执行遗传算法（添加早停机制和进度条）"""
        population = self.initialize_population()
        
        # 评估初始种群
        print("评估初始种群...")
        for individual in population:
            self.decode_and_evaluate(individual)
            
        best_individual = max(population, key=lambda x: x.fitness)
        best_fitness = best_individual.fitness
        generations_without_improvement = 0
        
        # 迭代进化
        initial_temperature = 1.0
        
        # 使用tqdm创建进度条
        pbar = tqdm(range(self.generations), desc="遗传算法进化中")
        try:
            for generation in pbar:
                temperature = initial_temperature * (1 - generation / self.generations)
                
                # 选择精英
                population.sort(key=lambda x: x.fitness, reverse=True)
                elites = [deepcopy(ind) for ind in population[:self.elite_size]]
                
                # 生成新一代
                offspring = self._generate_offspring(population, temperature)
                
                # 评估子代
                for individual in offspring:
                    self.decode_and_evaluate(individual)
                
                # 更新种群
                population = elites + offspring[:self.population_size - self.elite_size]
                
                # 更新最优解
                current_best = max(population, key=lambda x: x.fitness)
                if current_best.fitness > best_fitness:
                    best_individual = deepcopy(current_best)
                    best_fitness = current_best.fitness
                    generations_without_improvement = 0
                    # 更新进度条描述
                    pbar.set_description(f"最优适应度: {best_fitness:.4f}")
                else:
                    generations_without_improvement += 1
                
                # 更新进度条附加信息
                pbar.set_postfix({
                    'bins': best_individual.num_bins,
                    'no_improve': generations_without_improvement
                })
                
                # 早停检查
                if generations_without_improvement >= self.early_stop_generations:
                    pbar.write(f"提前在第 {generation + 1} 代停止，因为连续 {self.early_stop_generations} 代没有改进")
                    break
                    
        finally:
            pbar.close()
        
        # 使用最优解更新装箱结果
        self.bins = best_individual.bins
        return self.get_result()
        
    def _generate_offspring(self, population: List[Individual], temperature: float) -> List[Individual]:
        """生成子代（提取为独立方法以提高可读性）"""
        parents = self.select_parents(population)
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and self.rng.random() < self.crossover_rate:
                seq1, seq2 = self.order_crossover(
                    parents[i].packing_sequence,
                    parents[i+1].packing_sequence
                )
                strat1, strat2 = self.arithmetic_crossover(
                    parents[i].placement_strategy,
                    parents[i+1].placement_strategy
                )
                
                child1 = Individual(seq1, strat1)
                child2 = Individual(seq2, strat2)
                
                self.mutate(child1, temperature)
                self.mutate(child2, temperature)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([deepcopy(parents[i])])
                if i + 1 < len(parents):
                    offspring.extend([deepcopy(parents[i+1])])
        
        return offspring
