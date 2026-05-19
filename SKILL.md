# MH370 Pixel Hunter — 像素猎手

> 卫星影像上的十年追迹。人类与智能体的协作，一个**未完成**的项目。

多方向整合的 MH370 分析智能体。覆盖卫星影像筛查、多约束交叉定位、可检测性仿真、飞行路径分析四条路径的综合工具。

**当前状态：** 能力边界已到。缺高精度 SAR 影像、7次握手信号原始数据、MTSAT-2 全分辨率存档。
在此邀请全球极客和智能体一起往下走。

最值得搜索海域交汇点：**33.4°S, 99.8°E**（SE 印度洋）。

## 目录结构

```
MH370_Pixel_Hunter/
├── SKILL.md                          # 本文件
├── main.py                           # 统一 CLI 入口
├── skyprint/
│   ├── skyprint_agent.py             # SkyPrint 凝结尾迹筛查管道
│   └── contrail_detector.py          # Radon 变换线性特征检测器
├── modeling/
│   ├── multi_constraint_intersection.py  # 四约束交叉定位（碎片+弧线+燃油+雷达）
│   ├── mtsat2_detectability.py       # MTSAT-2 VIS 碎屑场可检测性仿真
│   └── multi_sat_strategy.py         # 多卫星联合覆盖策略
└── flight/
    ├── known_data.py                 # MH370 已知数据（事件、BTO/BFO、残骸）
    ├── bfo_analysis.py               # BFO 差分急动度分析
    ├── flight_path_monte_carlo.py    # 飞行路径蒙特卡洛
    ├── oscar_currents.py             # 洋流反向追踪
    └── seventh_arc.py                # 第7弧线几何
```

## 统一 CLI

```bash
# 卫星数据可用性检查（CMR API + ERDDAP + Worldview）
python main.py data --date 2014-03-08 --full
python main.py data --date 2014-03-08 --bbox 92,-40,104,-25 --full

# 凝结尾迹卫星筛查（仅近期数据可用）
python main.py skyprint \
    --bbox "31S,38S,92E,104E" \
    --date 2014-03-08 --window 7 \
    --output review_mh370/

# 多约束交叉定位
python main.py intersect \
    --samples 50000 --output intersection.png

# MTSAT-2 可检测性仿真
python main.py detectability \
    --lat -35.5 --lon 95.5 --output detect_output/

# 飞行路径分析
python main.py flight bfo
python main.py flight mc --samples 100000
```

## 四层鉴别器（skyprint 管道）

| 阶段 | 鉴别器 | 权重 | 物理基础 |
|------|--------|:----:|----------|
| Radon 粗筛 | CoarseScreener | — | 海面线性特征积累 |
| 3a | TextureEntropy | 0.15 | 梯度方向一致性（低熵=尾迹，测试无效） |
| 3b | ParallelTrack | 0.15 | 双发尾迹平行配对联（间距 1-5km） |
| 3d | CloudDiscriminator | 0.25 | 亮度阈值 + 局部对比度（真云 vs 尾迹） |
| — | ContrailScorer | 加权集成 | geometry(0.25)+texture+pairs+cloud+context |

## 预测精度（诚实评估）

| 路径 | 可靠性 | 理由 |
|------|:------:|------|
| 多约束交叉定位 | ★★★★☆ | 四约束交叉已验证收敛，峰值 33.4°S/99.8°E |
| 卫星影像尾迹检测 | ★★☆☆☆ | 搜索区 120,000 km² vs 尾迹 ~300m² |
| MTSAT-2 碎屑可检测 | ★★★☆☆ | F1=0.727 但 80% 云覆盖制约 |
| BFO 信号分析 | ★★★☆☆ | 可约束航向角，不能定位终点 |

找到确切坠机位置概率 < 0.5%（Fugro 已侧扫最可能区域）。项目本质是**方法论展示 + 开源工具**，非搜索承诺。

## 依赖

- Python ≥ 3.8
- numpy, scipy, matplotlib, scikit-image
- urllib (GIBS 拉取)

## License

MIT
