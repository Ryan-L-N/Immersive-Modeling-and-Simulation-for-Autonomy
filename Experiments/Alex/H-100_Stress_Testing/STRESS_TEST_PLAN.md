# H-100 Environment Scaling Stress Test

## Objective
Determine the maximum number of parallel environments the NVIDIA H100 NVL (96 GB VRAM) can support, find the optimal throughput sweet spot, and compare against a consumer-grade RTX 2000 Ada (8 GB VRAM).

---

## Hardware Under Test

| Spec | H100 Server | Local Laptop |
|---|---|---|
| **GPU** | NVIDIA H100 NVL | NVIDIA RTX 2000 Ada Generation |
| **VRAM** | 95,830 MiB (96 GB) | 8,188 MiB (8 GB) |
| **TDP** | 400W | 35W |
| **CPU** | Intel Xeon Platinum 8581V (60c/120t) | Intel Core Ultra 9 185H (16c/22t) |
| **RAM** | 1 TiB | 32 GB |
| **Driver** | 580.126.16 | 573.44 |
| **CUDA** | 13.0 | 12.8 |
| **Isaac Sim** | 5.1.0 | 5.1.0 |
| **Isaac Lab** | 0.54.3 | 0.54.2 |

---

## Test Configuration

| Parameter | Value |
|---|---|
| **Task** | Isaac-Velocity-Rough-Anymal-C-v0 |
| **RL Framework** | RSL-RL (PPO) |
| **Iterations per test** | 10 |
| **Mode** | Headless |

---

## Test Matrix

### H100 Tests
| Test # | Num Envs | Expected VRAM | Status |
|---|---|---|---|
| H1 | 1,024 | ~6 GB | ⬜ |
| H2 | 2,048 | ~10 GB | ⬜ |
| H3 | 4,096 | ~18 GB | ⬜ |
| H4 | 8,192 | ~34 GB | ⬜ |
| H5 | 16,384 | ~65 GB | ⬜ |
| H6 | 32,768 | ~90+ GB | ⬜ |
| H7 | 65,536 | OOM? | ⬜ |

### Local RTX 2000 Ada Tests
| Test # | Num Envs | Expected VRAM | Status |
|---|---|---|---|
| L1 | 1,024 | ~6 GB | ⬜ |
| L2 | 2,048 | ~7+ GB | ⬜ |
| L3 | 4,096 | OOM? | ⬜ |

---

## Metrics Collected Per Test

### GPU Metrics
| Metric | Source | Unit |
|---|---|---|
| GPU utilization | nvidia-smi (sampled every 1s) | % |
| GPU memory utilization | nvidia-smi (sampled every 1s) | % |
| VRAM used | nvidia-smi (sampled every 1s) | MiB |
| GPU temperature | nvidia-smi (sampled every 1s) | °C |
| GPU power draw | nvidia-smi (sampled every 1s) | W |
| GPU clock speed | nvidia-smi (sampled every 1s) | MHz |

### CPU Metrics
| Metric | Source | Unit |
|---|---|---|
| CPU utilization (overall) | mpstat (sampled every 1s) | % |
| CPU utilization (per-core) | mpstat (sampled every 1s) | % |
| RAM used | free (sampled every 1s) | MiB |
| Process CPU % | pidstat (sampled every 1s) | % |
| Process RSS memory | pidstat (sampled every 1s) | MiB |
| Load average | uptime | 1/5/15 min |

### Training Metrics
| Metric | Source | Unit |
|---|---|---|
| Steps/second | Training log | steps/s |
| Iteration time | Training log | seconds |
| Total training time | Training log | seconds |
| Timesteps per iteration | Training log | count |
| OOM (yes/no) | stderr | boolean |

---

## Execution Plan

### Phase 1: H100 Scaling Tests (~15 min)
Run each H100 test sequentially via automated script:
```bash
#!/bin/bash
# H100 Stress Test Script — Collects GPU + CPU metrics during training

conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
cd ~/IsaacLab

RESULTS_DIR=/home/t2user/stress_test_results
mkdir -p $RESULTS_DIR

for ENVS in 1024 2048 4096 8192 16384 32768 65536; do
    echo "============================================"
    echo "=== Testing $ENVS envs ==="
    echo "============================================"
    
    # Create per-test output directory
    TEST_DIR=$RESULTS_DIR/envs_${ENVS}
    mkdir -p $TEST_DIR
    
    # Record system baseline
    echo "--- Baseline ---" > $TEST_DIR/system_info.txt
    uptime >> $TEST_DIR/system_info.txt
    free -m >> $TEST_DIR/system_info.txt
    nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu,utilization.memory --format=csv >> $TEST_DIR/system_info.txt
    
    # Start GPU monitoring (sample every 1 second)
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm \
        --format=csv -l 1 > $TEST_DIR/gpu_metrics.csv 2>&1 &
    GPU_MONITOR_PID=$!
    
    # Start CPU monitoring (sample every 1 second)
    mpstat 1 > $TEST_DIR/cpu_metrics.txt 2>&1 &
    CPU_MONITOR_PID=$!
    
    # Start memory monitoring (sample every 1 second)
    while true; do
        echo "$(date +%Y-%m-%d_%H:%M:%S) $(free -m | grep Mem | awk '{print $2,$3,$4,$7}')" >> $TEST_DIR/ram_metrics.csv
        sleep 1
    done &
    RAM_MONITOR_PID=$!
    
    # Run training (10 iterations)
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task=Isaac-Velocity-Rough-Anymal-C-v0 \
        --headless \
        --max_iterations 10 \
        --num_envs $ENVS \
        > $TEST_DIR/training.log 2>&1
    
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE" >> $TEST_DIR/training.log
    
    # Stop all monitors
    kill $GPU_MONITOR_PID $CPU_MONITOR_PID $RAM_MONITOR_PID 2>/dev/null
    wait $GPU_MONITOR_PID $CPU_MONITOR_PID $RAM_MONITOR_PID 2>/dev/null
    
    # Record final state
    echo "--- Final ---" >> $TEST_DIR/system_info.txt
    uptime >> $TEST_DIR/system_info.txt
    free -m >> $TEST_DIR/system_info.txt
    nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu,utilization.memory --format=csv >> $TEST_DIR/system_info.txt
    
    # Extract summary metrics
    echo "envs=$ENVS" > $TEST_DIR/summary.txt
    echo "exit_code=$EXIT_CODE" >> $TEST_DIR/summary.txt
    grep "Computation:" $TEST_DIR/training.log | tail -1 >> $TEST_DIR/summary.txt
    grep "Training time:" $TEST_DIR/training.log >> $TEST_DIR/summary.txt
    grep "Iteration time:" $TEST_DIR/training.log | tail -1 >> $TEST_DIR/summary.txt
    
    # GPU peak metrics from csv
    echo "--- GPU Peaks ---" >> $TEST_DIR/summary.txt
    awk -F', ' 'NR>1 {if($2+0>max_util) max_util=$2+0; if($4+0>max_mem) max_mem=$4+0; if($5+0>max_temp) max_temp=$5+0; if($6+0>max_pwr) max_pwr=$6+0} END {print "peak_gpu_util="max_util"%"; print "peak_vram="max_mem"MiB"; print "peak_temp="max_temp"C"; print "peak_power="max_pwr"W"}' $TEST_DIR/gpu_metrics.csv >> $TEST_DIR/summary.txt 2>/dev/null
    
    # CPU peak from mpstat
    echo "--- CPU Peaks ---" >> $TEST_DIR/summary.txt
    awk '/all/ {if(100-$NF > max) max=100-$NF} END {print "peak_cpu_util="max"%"}' $TEST_DIR/cpu_metrics.txt >> $TEST_DIR/summary.txt 2>/dev/null
    
    # RAM peak
    echo "--- RAM Peaks ---" >> $TEST_DIR/summary.txt
    awk '{if($2+0>max) max=$2+0} END {print "peak_ram_used="max"MiB"}' $TEST_DIR/ram_metrics.csv >> $TEST_DIR/summary.txt 2>/dev/null
    
    echo "Test $ENVS envs complete (exit=$EXIT_CODE)"
    
    # Cool-down between tests
    sleep 10
done

# Generate combined results CSV
echo "envs,steps_per_sec,iter_time_s,training_time_s,peak_gpu_util,peak_vram_mib,peak_gpu_temp,peak_gpu_power,peak_cpu_util,peak_ram_mib,oom" > $RESULTS_DIR/combined_results.csv
for TEST_DIR in $RESULTS_DIR/envs_*; do
    ENVS=$(basename $TEST_DIR | sed 's/envs_//')
    SPS=$(grep "Computation:" $TEST_DIR/training.log 2>/dev/null | tail -1 | grep -oP '\d+ steps' | grep -oP '\d+' || echo "0")
    ITER_T=$(grep "Iteration time:" $TEST_DIR/training.log 2>/dev/null | tail -1 | grep -oP '[\d.]+s' | grep -oP '[\d.]+' || echo "0")
    TRAIN_T=$(grep "Training time:" $TEST_DIR/training.log 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    GPU_U=$(grep "peak_gpu_util" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    VRAM=$(grep "peak_vram" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    TEMP=$(grep "peak_temp" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    PWR=$(grep "peak_power" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    CPU_U=$(grep "peak_cpu_util" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    RAM=$(grep "peak_ram_used" $TEST_DIR/summary.txt 2>/dev/null | grep -oP '[\d.]+' || echo "0")
    OOM=$(grep -q "CUDA out of memory\|exit_code=137\|killed" $TEST_DIR/training.log 2>/dev/null && echo "YES" || echo "NO")
    echo "$ENVS,$SPS,$ITER_T,$TRAIN_T,$GPU_U,$VRAM,$TEMP,$PWR,$CPU_U,$RAM,$OOM" >> $RESULTS_DIR/combined_results.csv
done

echo "=== All tests complete ==="
echo "Results in: $RESULTS_DIR/combined_results.csv"
cat $RESULTS_DIR/combined_results.csv
```

### Phase 2: Local RTX 2000 Ada Tests (~5 min)
Same monitoring approach locally with env counts: 1024, 2048, 4096.

### Phase 3: Data Collection
Each test produces a directory with:
```
stress_test_results/
├── combined_results.csv          # One-line-per-test summary
├── envs_1024/
│   ├── training.log              # Full training output
│   ├── gpu_metrics.csv           # GPU util/VRAM/temp/power every 1s
│   ├── cpu_metrics.txt           # CPU util per-core every 1s
│   ├── ram_metrics.csv           # RAM usage every 1s
│   ├── system_info.txt           # Before/after system state
│   └── summary.txt               # Peak values extracted
├── envs_2048/
│   └── ...
└── ...
```

---

## Expected Results

### Throughput Curve (Projected)
```
Steps/s
  ^
  |                    ___________
  |                 __/           \___
  |              __/                  \___
  |           __/                         \___
  |        __/                                \__ (OOM)
  |     __/
  |  __/
  | /
  +-------------------------------------------------> Num Envs
    1K    2K    4K    8K   16K   32K   65K
```

- **Linear region** (1K-8K): Steps/s scales roughly linearly
- **Plateau** (8K-16K): GPU compute saturated, marginal gains
- **Decline/OOM** (32K+): Memory bottleneck, possible OOM

### VRAM Usage (Projected)
```
VRAM (GB)
  ^
96|                                          xxxxxx (OOM)
  |                                    xxxxx
  |                              xxxxx
  |                        xxxxx
  |                  xxxxx
  |            xxxxx
  |      xxxxx
  | xxxxx
  +-------------------------------------------------> Num Envs
    1K    2K    4K    8K   16K   32K   65K
```

---

## Results Table (To Be Filled)

### H100 Results
| Envs | Steps/s | Iter Time (s) | Training Time (s) | VRAM (MiB) | GPU Util (%) | Temp (°C) | Power (W) | OOM |
|---|---|---|---|---|---|---|---|---|
| 1,024 | — | — | — | — | — | — | — | — |
| 2,048 | — | — | — | — | — | — | — | — |
| 4,096 | — | — | — | — | — | — | — | — |
| 8,192 | — | — | — | — | — | — | — | — |
| 16,384 | — | — | — | — | — | — | — | — |
| 32,768 | — | — | — | — | — | — | — | — |
| 65,536 | — | — | — | — | — | — | — | — |

### Local RTX 2000 Ada Results
| Envs | Steps/s | Iter Time (s) | Training Time (s) | VRAM (MiB) | GPU Util (%) | Temp (°C) | Power (W) | OOM |
|---|---|---|---|---|---|---|---|---|
| 1,024 | — | — | — | — | — | — | — | — |
| 2,048 | — | — | — | — | — | — | — | — |
| 4,096 | — | — | — | — | — | — | — | — |

---

## Key Questions to Answer
1. **What is the max env count** the H100 can handle before OOM?
2. **What env count gives peak steps/s** (optimal throughput)?
3. **At what env count does the RTX 2000 Ada OOM?**
4. **What is the H100's speedup factor** over the RTX 2000 Ada at max shared env count?
5. **Does the H100 thermally throttle** under sustained load?
6. **What is the recommended env count** for production training runs?

---

## Notes
- Each test uses fresh process launch (no warm GPU state carryover)
- 10-second cooldown between tests
- Warp kernels are pre-cached from prior runs
- `libglu1-mesa` installed on H100 for headless rendering
- All tests run in headless mode (no GUI/display)
