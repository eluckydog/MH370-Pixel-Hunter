# MH370 Pixel Hunter — 已获取数据索引

## skyprint/ —— 凝结尾迹管道验证

| 文件 | 大小 | 来源 | 描述 |
|------|:----:|------|------|
| `validation_results.json` | 8KB | 管道输出 v2 | NAT 候选 18 条，平均置信度 0.558 |
| `validation_results_v3.json` | 8KB | 管道输出 v3 | NAT 候选 15 条，平均置信度 0.412 |
| `validation_neg_control_v3.json` | 8KB | 负对照输出 | 南大西洋负对照 15 候选，虚警率持平 |
| `validation_nat_vis.png` | 3MB | Radon 标注图 | v2 NAT 候选标注（含 FWHM 彩标） |
| `validation_nat_vis_v3.png` | 2.5MB | Radon 标注图 | v3 NAT 候选标注 |
| `validation_neg_control_v3.png` | 2.8MB | Radon 标注图 | 南大西洋负对照候选标注 |
| `ocean_mask_diag.png` | 1.8MB | 诊断图 | 海洋掩膜 + NDWI 可视化 |
| `neg_control_debug.png` | 13KB | 诊断图 | 负对照调试 |
| `worldview_2026-05-18_*.png` | 3.3MB | MODIS Terra | MODIS 测试图：北大西洋 (40°N, -30°W) |
| `worldview_20260518_north_atlantic.png` | 1.6MB | MODIS Terra | 北大西洋宽幅图 |

## modeling/ —— 建模结果

| 文件 | 大小 | 来源 | 描述 |
|------|:----:|------|------|
| `multi_constraint_intersection.png` | 443KB | 四约束交叉定位 | 峰值 33.4°S, 99.8°E |
| `intersection_detail.png` | 146KB | 同上 | 交集细节放大 |
| `mtsat2_detectability.png` | 434KB | MTSAT VIS 仿真 | 50×50 子像素网格对齐，F1=0.727 |
| `mtsat2_diurnal_images.png` | 135KB | 同上 | 日内多时相对比 |
| `debris_detection.png` | 3MB | 碎屑检测仿真 | 子像素碎片可视化 |
| `crash_sar_sim.png` | 2MB | SAR 仿真 | 碰撞点 SAR 模拟 |
| `sar_wake_simulation.png` | 2MB | SAR 仿真 | SAR 尾迹模拟 |
| `sar_wake_sim_v2.png` | 1.3MB | SAR 仿真 v2 | 改进版 |
| `gibs_3857.jpg` | 6MB | GIBS 地图 | EPSG:3857 测试 |
| `gibs_wms.jpg` | 11KB | GIBS WMS | WMS 测试 |
| `skyprint_test/` | 3 文件 | SKyPrint v1 | 完整管道运行周期 |
| `skyprint_test_v2/` | 3 文件 | SKyPrint v2 | 同上 |
| `skyprint_test_v3/` | 4 文件 | SKyPrint v3 + Cloud | 含云判别器 |

## satellite_coverage/ —— 卫星覆盖策略

| 文件 | 大小 | 描述 |
|------|:----:|------|
| `multi_sat_strategy.png` | 188KB | 多卫星联合覆盖图示 |
| `satellite_coverage.png` | 344KB | 覆盖图 |
| `sat_1.jpg`, `sat_2.jpg` | 6KB | 卫星测试图像 |
| `recent_test.jpg` | 6KB | 近期测试图像 |

## validation/ —— 验证对照

| 文件 | 大小 | 描述 |
|------|:----:|------|
| `terra_0802.jpg` ~ `aqua_0917.jpg` | 22KB/张 | MODIS Terra/Aqua 夜间过境图 ×8 |
| `neg_control_*.png` | ~90KB | 南太平洋负对照多位置尝试 |

## cache/ —— API 缓存

CMR、Worldview、ERDDAP 查询缓存，过期自动刷新。
