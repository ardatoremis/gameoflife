import random
import copy
import math
import time
import statistics # Ortalama ve standart sapma
import matplotlib.pyplot as plt # Grafik çizimi

# --- Izgara Print Fonksiyonu ---
def print_grid(grid, grid_size_x, grid_size_y):
    """Izgarayı konsola yazdırır."""
    if not grid or not grid[0]:
        print("Grid is empty or invalid.")
        return
    print("-" * (grid_size_y * 2 + 1))
    for r in range(grid_size_x):
        row_str = "|".join(map(str, grid[r][:grid_size_y]))
        print(f"|{row_str}|")
    print("-" * (grid_size_y * 2 + 1))

# --- Game of Life Simülasyon Sınıfı ---
class GameOfLife:
    def __init__(self, grid_size_x, grid_size_y):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid = None # Başlangıç gridi dışarıdan verilecek

    def set_grid(self, new_grid):
        """Izgarayı ayarlar (derin kopya ile)."""
        if not new_grid or not new_grid[0] or len(new_grid) != self.grid_size_x or len(new_grid[0]) != self.grid_size_y:
             raise ValueError(f"Provided grid dimensions ({len(new_grid)}x{len(new_grid[0]) if new_grid and new_grid[0] else 0}) do not match GameOfLife instance ({self.grid_size_x}x{self.grid_size_y}).")
        self.grid = copy.deepcopy(new_grid)


    def count_live_neighbors(self, x, y):
        """Belirli bir hücrenin canlı komşularını sayar."""
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                # Sınırların dışını ölü kabul et
                if 0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y:
                    # Grid'in None olup olmadığını kontrol et
                    if self.grid and self.grid[nx] and ny < len(self.grid[nx]):
                         count += self.grid[nx][ny]
        return count

    def step(self):
        """Simülasyonun bir adımını ilerletir."""
        if self.grid is None:
            raise ValueError("Grid is not set before calling step().")

        new_grid = [[0 for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                live_neighbors = self.count_live_neighbors(x, y)
                # GoL Kuralları
                if self.grid[x][y] == 1: # Canlı hücre
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_grid[x][y] = 0 # Ölür (yalnızlık veya kalabalık)
                    else:
                        new_grid[x][y] = 1 # Yaşar (2 veya 3 komşu)
                else: # Ölü hücre
                    if live_neighbors == 3:
                        new_grid[x][y] = 1 # Canlanır (tam 3 komşu)
                    else:
                        new_grid[x][y] = 0 # Ölü kalır
        self.grid = new_grid

    def run_simulation(self, initial_grid, steps):
        """
        Belirli bir başlangıç gridinden başlayarak simülasyonu çalıştırır
        ve son grid durumunu döndürür. Orijinal gridi değiştirmez.
        """
        # Başlangıç gridi boyutlarını kontrol et
        if not initial_grid or not initial_grid[0] or len(initial_grid) != self.grid_size_x or len(initial_grid[0]) != self.grid_size_y:
             raise ValueError("Invalid initial_grid dimensions provided to run_simulation.")

        temp_gol = GameOfLife(self.grid_size_x, self.grid_size_y)
        temp_gol.set_grid(initial_grid) # Başlangıç gridini kopyala
        for _ in range(steps):
            temp_gol.step()
        return temp_gol.grid # Son grid durumunu döndür

    def count_live_cells(self, specific_grid=None):
        """Belirli bir griddeki veya mevcut griddeki canlı hücreleri sayar."""
        grid_to_count = specific_grid if specific_grid else self.grid
        if grid_to_count is None:
            return 0
        count = 0
        # Boyutları doğrula
        if len(grid_to_count) != self.grid_size_x or (self.grid_size_x > 0 and len(grid_to_count[0]) != self.grid_size_y):
             print(f"Warning: Grid dimensions mismatch in count_live_cells. Expected {self.grid_size_x}x{self.grid_size_y}, got {len(grid_to_count)}x{len(grid_to_count[0]) if grid_to_count and grid_to_count[0] else 0}")
             return 0 # Hatalı boyut durumunda 0 döndür

        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                count += grid_to_count[x][y]
        return count


    def calculate_fitness(self, initial_grid, target_step):
        """
        Bir başlangıç konfigürasyonunun uygunluğunu hesaplar:
        target_step adım sonraki canlı hücre sayısı.
        """
        try:
            final_grid = self.run_simulation(initial_grid, target_step)
            return self.count_live_cells(final_grid)
        except ValueError as e:
            print(f"Error during fitness calculation: {e}")
            print("Problematic initial_grid:")
            # print_grid(initial_grid, self.grid_size_x, self.grid_size_y)
            return -1 # Hata durumunda negatif fitness


# --- Genetik Algoritma Sınıfı ---
class GeneticAlgorithm:
    def __init__(self, gol_simulator, population_size, grid_size_x, grid_size_y, target_step, generations, pc, pm, tournament_size=3):
        self.simulator = gol_simulator
        self.population_size = population_size
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.target_step = target_step
        self.generations = generations
        self.pc = pc # Crossover probability
        self.pm = pm # Mutation probability
        self.tournament_size = tournament_size
        self.population = self._initialize_population()
        self.fitness_scores = [0] * population_size
        self.best_individual = None
        self.best_fitness = -1
        self.fitness_history = [] # Fitness geçmişini tutacak liste

    def _initialize_population(self):
        pop = []
        for _ in range(self.population_size):
            grid = [[random.randint(0, 1) for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
            pop.append(grid)
        return pop

    def evaluate_population(self):
        current_gen_best_fitness = -1 # Bu jenerasyonun en iyisi
        for i, individual in enumerate(self.population):
            # Hatalı birey kontrolü
            if not individual or not individual[0] or len(individual) != self.grid_size_x or len(individual[0]) != self.grid_size_y:
                 print(f"Warning: Invalid individual found in population at index {i}. Skipping.")
                 self.fitness_scores[i] = -1
                 continue
            try:
                fitness = self.simulator.calculate_fitness(individual, self.target_step)
                self.fitness_scores[i] = fitness
                if fitness > current_gen_best_fitness:
                    current_gen_best_fitness = fitness
                # Genel en iyiyi güncelle
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = copy.deepcopy(individual)
            except Exception as e:
                 print(f"Error evaluating individual {i}: {e}")
                 self.fitness_scores[i] = -1 # Hata durumunda fitness'ı -1 yap

        # Her jenerasyon sonunda o ana kadarki en iyi fitness'ı kaydet
        self.fitness_history.append(self.best_fitness)


    def selection(self):
        selected_parents = []
        # Fitness'ı -1 olanları (hatalıları) ayıkla
        valid_population_with_fitness = [(ind, fit) for ind, fit in zip(self.population, self.fitness_scores) if fit != -1]

        if not valid_population_with_fitness: # Eğer geçerli birey kalmadıysa
             print("Warning: No valid individuals left for selection.")
             return self._initialize_population() # return []

        # Seçilecek ebeveyn sayısını popülasyon boyutuna tamamla
        num_parents_to_select = self.population_size

        for _ in range(num_parents_to_select):
             # Turnuva için yeterli birey varsa sample kullan, yoksa olanlardan seç (replace=True)
            k = min(self.tournament_size, len(valid_population_with_fitness))
            if len(valid_population_with_fitness) >= k:
                 tournament = random.sample(valid_population_with_fitness, k)
            else:
                 # Eğer turnuva boyutu kadar bile birey yoksa, olanları tekrar tekrar seçebilir
                 tournament = random.choices(valid_population_with_fitness, k=k)

            if not tournament: continue # Turnuva boşsa atla

            winner = max(tournament, key=lambda item: item[1])[0]
            selected_parents.append(copy.deepcopy(winner))

        # Eğer yeterli ebeveyn seçilemediyse rastgele doldur
        while len(selected_parents) < self.population_size:
            selected_parents.append(copy.deepcopy(random.choice(valid_population_with_fitness)[0]))

        return selected_parents


    def crossover_individual(self, parent1, parent2):
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        if random.random() < self.pc:

            if not parent1 or not parent1[0] or not parent2 or not parent2[0]:
                 print("Warning: Invalid parents for crossover.")
                 return parent1, parent2 # Değişiklik yapmadan döndür

            rows, cols = self.grid_size_x, self.grid_size_y
            if rows <= 0 or cols <= 0: return child1, child2 # Geçersiz boyutlar

            r1 = random.randint(0, rows - 1)
            r2 = random.randint(r1, rows - 1)
            c1 = random.randint(0, cols - 1)
            c2 = random.randint(c1, cols - 1)

            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    # crossover 2d
                    try: # Indeks hatalarını yakala
                       child1[r][c], child2[r][c] = child2[r][c], child1[r][c]
                    except IndexError:
                       print(f"Warning: Index error during crossover at ({r},{c}). Skipping swap.")
                       continue # Hata olursa bu hücreyi atla

        return child1, child2

    def mutate_individual(self, individual):
        # Boyut kontrolü
        if not individual or not individual[0] or len(individual) != self.grid_size_x or len(individual[0]) != self.grid_size_y:
             print("Warning: Invalid individual passed to mutation.")
             return individual # Değişiklik yapmadan döndür

        mutated_individual = copy.deepcopy(individual)
        for r in range(self.grid_size_x):
            for c in range(self.grid_size_y):
                if random.random() < self.pm:
                    mutated_individual[r][c] = 1 - mutated_individual[r][c]
        return mutated_individual

    def evolve(self):
        self.evaluate_population() # Değerlendir ve fitness geçmişini güncelle

        parents = self.selection()
        next_population = []

        if not parents: # Eğer ebeveyn seçilemediyse
             print("Error: No parents selected. Cannot evolve. Re-initializing population.")
             self.population = self._initialize_population()
             self.best_fitness = -1 # En iyiyi sıfırla
             self.fitness_history = [] # Geçmişi sıfırla
             return # Evrimleşme adımını atla

        # Elitizm
        if self.best_individual and self.best_fitness != -1:
             # Elit bireyin boyutlarını kontrol et
             if len(self.best_individual) == self.grid_size_x and len(self.best_individual[0]) == self.grid_size_y:
                  next_population.append(copy.deepcopy(self.best_individual))
             else:
                   print("Warning: Best individual has incorrect dimensions, skipping elitism.")


        while len(next_population) < self.population_size:
            if len(parents) < 2: # Çaprazlama için yeterli ebeveyn yoksa
                 if parents: # En az bir ebeveyn varsa onu kopyala
                      p1 = copy.deepcopy(parents[0])
                      child1 = self.mutate_individual(p1)
                      next_population.append(child1)
                      if len(next_population) == self.population_size: break
                      # İkinci çocuk yerine başka bir mutasyonlu kopya
                      child2 = self.mutate_individual(copy.deepcopy(parents[0]))
                      next_population.append(child2)
                 else: # Hiç ebeveyn yoksa döngüden çık
                      break
            else: # Normal çaprazlama
                 p1, p2 = random.sample(parents, 2)
                 child1, child2 = self.crossover_individual(p1, p2)
                 child1 = self.mutate_individual(child1)
                 child2 = self.mutate_individual(child2)

                 # Çocukların boyutlarını kontrol et
                 if child1 and child1[0] and len(child1) == self.grid_size_x and len(child1[0]) == self.grid_size_y:
                     next_population.append(child1)
                 else: print("Warning: Invalid child1 skipped.")

                 if len(next_population) < self.population_size:
                      if child2 and child2[0] and len(child2) == self.grid_size_x and len(child2[0]) == self.grid_size_y:
                          next_population.append(child2)
                      else: print("Warning: Invalid child2 skipped.")


        # Popülasyonu tamamlama (eğer yukarıdaki kontroller nedeniyle eksik kaldıysa)
        while len(next_population) < self.population_size:
             print(f"Warning: Population size ({len(next_population)}) is less than expected ({self.population_size}). Adding random individual.")
             grid = [[random.randint(0, 1) for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
             next_population.append(grid)


        self.population = next_population[:self.population_size]


    def run(self):
        # Başlangıçta geçmişi ve en iyiyi sıfırlıyoruz
        self.fitness_history = []
        self.best_fitness = -1
        self.best_individual = self._initialize_population()[0] # Rastgele bir başlangıç ata

        start_time = time.time()
        print(f"--- Running Genetic Algorithm (Pop:{self.population_size}, Gen:{self.generations}, Pc:{self.pc}, Pm:{self.pm}) ---")

        # İlk popülasyonu değerlendirerek başla
        self.evaluate_population()
        print(f"Generation 0 | Initial Best Fitness: {self.best_fitness}")

        for gen in range(self.generations):
            self.evolve()
            # Geri bildirim sıklığı her 10 jenerasyonda bir
            if (gen + 1) % 10 == 0 or gen == self.generations - 1:
                 print(f"Generation {gen+1}/{self.generations} | Overall Best Fitness: {self.best_fitness}")

        # Son nesli tekrar değerlendirmeye gerek yok, evolve içinde yapılıyor.
        end_time = time.time()
        run_time = end_time - start_time
        print(f"GA Finished in {run_time:.2f} seconds.")
        print(f"Best Fitness Found: {self.best_fitness}")
        # print("Best Initial Grid Found by GA:")
        # print_grid(self.best_individual, self.grid_size_x, self.grid_size_y)

        # Fitness geçmişini de döndür
        return self.best_individual, self.best_fitness, run_time, self.fitness_history


# --- Hill Climbing Sınıfı ---
class HillClimbing:
    def __init__(self, gol_simulator, grid_size_x, grid_size_y, target_step, max_iterations):
        self.simulator = gol_simulator
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.target_step = target_step
        self.max_iterations = max_iterations
        self.current_grid = None
        self.current_fitness = -1
        self.fitness_history = []

    def _initialize_state(self):
        self.current_grid = [[random.randint(0, 1) for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
        self.current_fitness = self.simulator.calculate_fitness(self.current_grid, self.target_step)
        self.fitness_history = [self.current_fitness] # Başlangıç fitness'ını ekle

    def get_neighbors(self, grid):
        neighbors = []
        if not grid or not grid[0]: return neighbors # Güvenlik kontrolü
        for r in range(self.grid_size_x):
            for c in range(self.grid_size_y):
                neighbor_grid = copy.deepcopy(grid)
                neighbor_grid[r][c] = 1 - neighbor_grid[r][c]
                neighbors.append(neighbor_grid)
        return neighbors

    def run(self):
        start_time = time.time()
        self._initialize_state()
        print(f"--- Running Hill Climbing (MaxIter:{self.max_iterations}) ---")
        print(f"Initial Random Fitness: {self.current_fitness}")

        for iteration in range(self.max_iterations):
            neighbors = self.get_neighbors(self.current_grid)
            if not neighbors: break

            best_neighbor_fitness = -1
            best_neighbor_grid = None
            try:
                # Komşuların fitness'larını hesaplıyoruz
                 neighbor_fitness_pairs = []
                 for n in neighbors:
                      fit = self.simulator.calculate_fitness(n, self.target_step)
                      neighbor_fitness_pairs.append((fit, n))

                 if not neighbor_fitness_pairs: # Eğer fitness hesaplanamadıysa
                      print("Warning: Could not calculate fitness for any neighbor.")
                      break

                 best_neighbor_fitness, best_neighbor_grid = max(neighbor_fitness_pairs, key=lambda item: item[0])

            except Exception as e:
                 print(f"Error evaluating neighbors in HC iteration {iteration+1}: {e}")
                 break # Hata durumunda devam etme


            if best_neighbor_fitness > self.current_fitness:
                self.current_grid = best_neighbor_grid
                self.current_fitness = best_neighbor_fitness
                self.fitness_history.append(self.current_fitness) # İyileşme oldukça ekle
                # print(f"Iteration {iteration+1}: Found better neighbor. Fitness = {self.current_fitness}") # İsteğe bağlı detaylı log
            else:
                print(f"Iteration {iteration+1}: No better neighbor found. Stopping at local optimum.")
                break
        else:
             print(f"Reached max iterations ({self.max_iterations}).")

        end_time = time.time()
        run_time = end_time - start_time
        print(f"HC Finished in {run_time:.2f} seconds.")
        print(f"Best Fitness Found: {self.current_fitness}")
        # print("Best Initial Grid Found by HC:")
        # print_grid(self.current_grid, self.grid_size_x, self.grid_size_y)

        # Fitness geçmişini döndür
        return self.current_grid, self.current_fitness, run_time, self.fitness_history

# --- Simulated Annealing Sınıfı ---
class SimulatedAnnealing:
    def __init__(self, gol_simulator, grid_size_x, grid_size_y, target_step, initial_temp, cooling_rate, steps_per_temp, min_temp=0.1):
        self.simulator = gol_simulator
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.target_step = target_step
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.min_temp = min_temp
        self.current_grid = None
        self.current_fitness = -1
        self.best_grid = None
        self.best_fitness = -1
        self.fitness_history = []

    def _initialize_state(self):
        self.current_grid = [[random.randint(0, 1) for _ in range(self.grid_size_y)] for _ in range(self.grid_size_x)]
        self.current_fitness = self.simulator.calculate_fitness(self.current_grid, self.target_step)
        self.best_grid = copy.deepcopy(self.current_grid)
        self.best_fitness = self.current_fitness
        self.fitness_history = [self.best_fitness] # Başlangıçtaki en iyiyi ekle

    def get_random_neighbor(self, grid):
        neighbor_grid = copy.deepcopy(grid)
        if not grid or not grid[0] : return neighbor_grid # Güvenlik
        r = random.randint(0, self.grid_size_x - 1)
        c = random.randint(0, self.grid_size_y - 1)
        neighbor_grid[r][c] = 1 - neighbor_grid[r][c]
        return neighbor_grid

    def acceptance_probability(self, old_fitness, new_fitness, temperature):
        if new_fitness > old_fitness:
            return 1.0
        if temperature <= 1e-9: # Çok küçük sıcaklıklarda sıfır kabul et
            return 0.0
        try:
            # SA
            delta_e = new_fitness - old_fitness
            # Çok büyük negatif üsleri önlemek için -700 ile sınırladım
            exponent = max(delta_e / temperature, -700)
            prob = math.exp(exponent)
            return prob
        except OverflowError:
             return 0.0

    def run(self):
        start_time = time.time()
        try:
             self._initialize_state()
        except Exception as e:
             print(f"Error during SA initialization: {e}")
             return None, -1, 0, [] # Hata durumunda boş dön

        print(f"--- Running Simulated Annealing (T0:{self.initial_temp}, Cool:{self.cooling_rate}, Steps:{self.steps_per_temp}) ---")
        print(f"Initial Random Fitness: {self.current_fitness}")

        temperature = self.initial_temp
        iteration = 0

        while temperature > self.min_temp:
            steps_done_this_temp = 0
            while steps_done_this_temp < self.steps_per_temp:
                iteration += 1
                neighbor_grid = self.get_random_neighbor(self.current_grid)
                try:
                    neighbor_fitness = self.simulator.calculate_fitness(neighbor_grid, self.target_step)
                except Exception as e:
                    print(f"Error calculating fitness for SA neighbor at iter {iteration}: {e}")
                    steps_done_this_temp += 1
                    continue # Bu komşuyu atla

                prob = self.acceptance_probability(self.current_fitness, neighbor_fitness, temperature)

                if prob > random.random():
                    self.current_grid = neighbor_grid
                    self.current_fitness = neighbor_fitness

                    if self.current_fitness > self.best_fitness:
                        self.best_fitness = self.current_fitness
                        self.best_grid = copy.deepcopy(self.current_grid)
                steps_done_this_temp += 1

            # Her sıcaklık adımı sonunda en iyi fitness'ı kaydet
            self.fitness_history.append(self.best_fitness)
            temperature *= self.cooling_rate
            # print(f"Temp: {temperature:.2f}, Best Fitness So Far: {self.best_fitness}") # İsteğe bağlı log

        end_time = time.time()
        run_time = end_time - start_time
        print(f"SA Finished in {run_time:.2f} seconds. Total iterations: {iteration}")
        print(f"Best Fitness Found: {self.best_fitness}")
        # print("Best Initial Grid Found by SA:")
        # print_grid(self.best_grid, self.grid_size_x, self.grid_size_y)

        # Fitness geçmişini döndür
        return self.best_grid, self.best_fitness, run_time, self.fitness_history


# --- Grafik Çizdirme Fonksiyonu ---
def plot_fitness_history(history_dict, title="Fitness History"):
    """
    Verilen fitness geçmişlerini çizer.
    history_dict: {'Algoritma Adı': [fitness_listesi], ...}
    """
    plt.figure(figsize=(12, 6))
    for label, history in history_dict.items():
        # X ekseni: Jenerasyon/İterasyon/Sıcaklık Adımı
        steps = range(len(history))
        plt.plot(steps, history, label=label, marker='.', linestyle='-')

    plt.title(title)
    plt.xlabel("Step (Generation/Improvement Iteration/Temperature Step)")
    plt.ylabel("Best Fitness Found So Far")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # safe_title = "".join(c if c.isalnum() else "_" for c in title)
    # plt.savefig(f"{safe_title}.png")
    plt.show()

# --- Ana Çalıştırma Bloğu (Deney Yönetimi ile) ---
if __name__ == "__main__":
    NUM_RUNS = 3 # Her konfigürasyonu kaç kere run'layacağız

    # -- Deney Senaryoları Tanımla --
    experiment_scenarios = [
        {"id": "5x5_T10", "grid_size_x": 5, "grid_size_y": 5, "target_step": 10},
        {"id": "10x10_T10", "grid_size_x": 10, "grid_size_y": 10, "target_step": 10},
        # {"id": "10x10_T20", "grid_size_x": 10, "grid_size_y": 10, "target_step": 20}, # Daha uzun sürebilir
        # {"id": "15x15_T10", "grid_size_x": 15, "grid_size_y": 15, "target_step": 10}, # Çok daha uzun sürebilir
    ]

    # -- Algoritma Parametre Setleri Tanımla --
    ga_param_sets = [
        {"id": "GA_P50_G50_PM05", "population_size": 50, "generations": 50, "pc": 0.8, "pm": 0.05, "tournament_size": 5},
        {"id": "GA_P100_G100_PM02", "population_size": 100, "generations": 100, "pc": 0.8, "pm": 0.02, "tournament_size": 5},
    ]
    hc_param_sets = [
        {"id": "HC_Iter200", "max_iterations": 200},
        {"id": "HC_Iter500", "max_iterations": 500},
    ]
    sa_param_sets = [
        {"id": "SA_T100_C97_S50", "initial_temp": 100.0, "cooling_rate": 0.97, "steps_per_temp": 50, "min_temp": 0.1},
        {"id": "SA_T500_C99_S100", "initial_temp": 500.0, "cooling_rate": 0.99, "steps_per_temp": 100, "min_temp": 0.1},
    ]

    # -- Deneyleri Çalıştır ve Sonuçları Sakla --
    all_results = {} # { 'scenario_id': { 'algo_param_id': {'fitnesses':[], 'times':[], 'histories':[]} } }

    for scenario in experiment_scenarios:
        scenario_id = scenario["id"]
        gsx, gsy, ts = scenario["grid_size_x"], scenario["grid_size_y"], scenario["target_step"]
        print(f"\n===== Starting Scenario: {scenario_id} (Grid: {gsx}x{gsy}, Target Step: {ts}) =====")
        all_results[scenario_id] = {}

        # Simülatörü her senaryo için bir kez oluşturuyoruz
        gol_simulator = GameOfLife(gsx, gsy)

        # --- GA Deneyleri ---
        for params in ga_param_sets:
            param_id = params["id"]
            print(f"\n--- Running {param_id} for {NUM_RUNS} times ---")
            results = {'fitnesses': [], 'times': [], 'histories': []}
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}")
                ga = GeneticAlgorithm(gol_simulator, params["population_size"], gsx, gsy, ts,
                                      params["generations"], params["pc"], params["pm"], params["tournament_size"])
                _, fitness, run_time, history = ga.run()
                results['fitnesses'].append(fitness)
                results['times'].append(run_time)
                results['histories'].append(history) # Tüm geçmişleri sakla
            all_results[scenario_id][param_id] = results

        # --- HC Deneyleri ---
        for params in hc_param_sets:
            param_id = params["id"]
            print(f"\n--- Running {param_id} for {NUM_RUNS} times ---")
            results = {'fitnesses': [], 'times': [], 'histories': []}
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}")
                hc = HillClimbing(gol_simulator, gsx, gsy, ts, params["max_iterations"])
                _, fitness, run_time, history = hc.run()
                results['fitnesses'].append(fitness)
                results['times'].append(run_time)
                results['histories'].append(history)
            all_results[scenario_id][param_id] = results

        # --- SA Deneyleri ---
        for params in sa_param_sets:
            param_id = params["id"]
            print(f"\n--- Running {param_id} for {NUM_RUNS} times ---")
            results = {'fitnesses': [], 'times': [], 'histories': []}
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}")
                sa = SimulatedAnnealing(gol_simulator, gsx, gsy, ts, params["initial_temp"],
                                        params["cooling_rate"], params["steps_per_temp"], params["min_temp"])
                _, fitness, run_time, history = sa.run()

                if fitness != -1:
                    results['fitnesses'].append(fitness)
                    results['times'].append(run_time)
                    results['histories'].append(history)
                else:
                    print(f"Run {i+1} failed for {param_id}, skipping results.")
            # Başarılı çalıştırma yoksa boş bırakma
            if not results['fitnesses']:
                 print(f"Warning: No successful runs recorded for {param_id} in scenario {scenario_id}")
                 # Boş veya varsayılan değerlerle doldurabiliriz veya atlayabiliriz
                 all_results[scenario_id][param_id] = {'fitnesses': [-1], 'times': [0], 'histories': [[]]}
            else:
                 all_results[scenario_id][param_id] = results


    # -- Sonuçları Özetle ve Yazdır --
    print("\n\n======= EXPERIMENT RESULTS SUMMARY =======")
    for scenario_id, scenario_data in all_results.items():
        print(f"\n----- Scenario: {scenario_id} -----")
        print("-" * (len(scenario_id) + 20))
        # Tablo başlığı
        print("| Algorithm Configuration          | Avg Fitness | Best Fitness | Avg Time (s) | Std Dev Fitness |")
        print("| :------------------------------- | :---------- | :----------- | :----------- | :-------------- |")

        for param_id, results in scenario_data.items():
             # Fitness -1 olanları (hatalıları) filtrele
            valid_fitnesses = [f for f in results['fitnesses'] if f != -1]
            valid_times = [t for f, t in zip(results['fitnesses'], results['times']) if f != -1]

            if valid_fitnesses: # Sadece geçerli sonuç varsa istatistik hesapla
                avg_fitness = statistics.mean(valid_fitnesses)
                best_fitness = max(valid_fitnesses)
                avg_time = statistics.mean(valid_times) if valid_times else 0
                # Standart sapma için en az 2 veri noktası lazım
                std_dev_fitness = statistics.stdev(valid_fitnesses) if len(valid_fitnesses) > 1 else 0.0
                print(f"| {param_id:<32} | {avg_fitness:^11.2f} | {best_fitness:^12} | {avg_time:^12.2f} | {std_dev_fitness:^15.2f} |")
            else: # Geçerli sonuç yoksa
                 print(f"| {param_id:<32} | {'N/A':^11} | {'N/A':^12} | {'N/A':^12} | {'N/A':^15} |")

    # -- Yakınsama Grafiği Çizdirme --
    # İlk senaryonun tüm algoritmalarının ilk çalıştırma geçmişleri
    example_scenario_id = experiment_scenarios[0]['id']
    if example_scenario_id in all_results:
        histories_to_plot = {}
        for param_id, results in all_results[example_scenario_id].items():
            if results['histories']: # Geçmiş boş değilse
                # İlk çalıştırmanın geçmişini al
                histories_to_plot[param_id] = results['histories'][0]

        if histories_to_plot:
            plot_fitness_history(histories_to_plot, title=f"Fitness Convergence for Scenario: {example_scenario_id} (Run 1)")
        else:
            print(f"\nNo valid histories found to plot for scenario {example_scenario_id}")