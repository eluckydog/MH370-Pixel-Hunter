# MH370 Pixel Hunter — 像素猎手

> 卫星影像上的十年追迹。人类与智能体的协作，一个**未完成**的项目。

🌐 [中文](#中文) · [EN](#english) · [FR](#français) · [RU](#русский) · [ES](#español) · [العربية](#العربية)

---

## 中文

这是一个**未完成**的项目。不是因为没有努力，是因为我们够不到那些数据。

MH370 消失已有十二年。本项目的初衷是以开源、低成本的方式，利用公开可用的卫星数据和计算资源，尝试缩小搜索范围。我们走到了能力边界——在此诚邀全世界的**极客、数据分析师、智能体开发者**一起往前走。

### 我们做了什么

| 路径 | 工具 | 结果 |
|------|------|------|
| **凝结尾迹筛查** | MODIS/VIIRS → Radon 变换 → 多鉴别器评分 | 管道可运行，但 2014 年数据需下载原始 HDF 自行渲染 |
| **多约束交叉定位** | 碎片洋流反推 + 第7弧线 + 燃油航程 + 雷达路径 | **最值得搜索区域已收敛** |
| **MTSAT-2 可检测性** | 子像素面积加权仿真，峰值 F1=0.727 | VIS 波段时间窗口 UTC 03:00-09:00 |
| **飞行路径分析** | BFO 差分急动度 + 蒙特卡洛 + 洋流反推 | 端到端可运行 |

### 🎯 最值得搜索的海域

**峰值交集坐标：33.4°S, 99.8°E（SE 印度洋）**

四个独立约束的交叉点——与 ATSB 官方搜索区方向高度一致。精度有待提高，但方向已清晰。

### 我们卡在哪里

| 缺失数据 | 为什么需要 | 谁能提供 |
|----------|-----------|---------|
| **7 次卫星-飞机握手信号的精确时序和BFO值** | 重新计算第7弧线，缩小至±5km | Inmarsat / 马来西亚当局 |
| **2014-03-08 高分辨率 SAR 影像** (Sentinel-1A, RADARSAT-2) | 海洋残骸检测，1-20米分辨率 | ESA / MDA / 各国航天局 |
| **MODIS/VIIRS L1B 热红外波段原始 HDF** | 2014 年原始数据（NASA LAADS DAAC 可下载，需处理管线） | 任何有计算资源的人 |
| **海底地形和洋流高精度模型** | 提高碎屑漂移反推精度 | 海洋学研究机构 |
| **MTSAT-2 / Himawari 全分辨率可见光存档** | 事发当天逐30分钟影像 | JMA / KMA |

**这些数据超出了我们的能力范围。但也许没有超出你的。**

### 开源邀请

我们在此**邀请全世界的极客和智能体**一起协作。

```bash
git clone https://github.com/.../MH370-Pixel-Hunter
pip install numpy scipy matplotlib scikit-image urllib3

python main.py data --date 2014-03-08
python main.py intersect --samples 100000
python main.py detectability --lat -33.4 --lon 99.8
python main.py flight bfo
python main.py skyprint --bbox "30S,40S,92E,104E" --date 2014-03-08
```

**如果你是智能体**：可以把本仓库作为 Tool/Skill 加载，直接调用 API 检查卫星覆盖，分析可检测性，迭代交叉定位模型。

**如果你有数据**：`data_sources.py` 已集成 NASA CMR API（无需注册即可发现 MODIS/VIIRS 历史 granule）。如需加工全分辨率数据，提交 Issue 或 PR。

### 🙏 致谢与请求

本项目是人类+智能体团队在有限公开数据下的探索。如果最终我们的工作**为破解 MH370 失踪谜团提供了任何有效的帮助**，希望破案人在公布成果时可以提及我们的工作。

我们相信，一旦这个项目成功，其最终影响不会限于 MH370——它将大大提高未来类似案例的搜救效率，为全人类造福。

---

## English

> A decade-long hunt on satellite imagery. A human-AI collaboration, an **unfinished** project.

This is an **unfinished** project. Not for lack of effort — we simply cannot reach the data we need.

MH370 disappeared twelve years ago. This project was born out of an attempt to narrow the search area using open-source, low-cost methods with publicly available satellite data and computing resources. We have reached the limits of our capabilities. Now we invite **geeks, data analysts, and AI agents worldwide** to carry this forward.

### What We Did

| Path | Tooling | Results |
|------|---------|---------|
| **Contrail screening** | MODIS/VIIRS → Radon transform → multi-discriminator scoring | Pipeline works, but 2014 data requires raw HDF download and rendering |
| **Multi-constraint intersection** | Debris drift + 7th arc + fuel range + radar path | **Search zone converged** |
| **MTSAT-2 detectability** | Sub-pixel area-weighted simulation, peak F1=0.727 | VIS window UTC 03:00-09:00 |
| **Flight path analysis** | BFO differential jerk + Monte Carlo + current backtracking | End-to-end operational |

### 🎯 Most Promising Search Area

**Intersection peak: 33.4°S, 99.8°E (SE Indian Ocean)**

Four independent constraints converge here — consistent with the ATSB official search zone. Precision can be improved, but the direction is clear.

### Where We're Stuck

| Missing Data | Why We Need It | Who Can Provide |
|-------------|----------------|-----------------|
| **Precise timing and BFO values of 7 satellite-aircraft handshakes** | Recompute 7th arc, constrain to ±5km | Inmarsat / Malaysian authorities |
| **2014-03-08 high-resolution SAR imagery** (Sentinel-1A, RADARSAT-2) | Sub-20m ocean debris detection | ESA / MDA / National space agencies |
| **MODIS/VIIRS L1B thermal IR raw HDF files** | Raw 2014 data (downloadable from NASA LAADS DAAC, needs processing pipeline) | Anyone with compute resources |
| **High-resolution bathymetry and ocean current models** | Improve debris drift backtracking | Oceanographic research institutes |
| **MTSAT-2 / Himawari full-resolution visible archive** | 30-minute interval imagery for the day of disappearance | JMA / KMA |

**These data are beyond our reach. But perhaps not beyond yours.**

### Open Invitation

We invite **geeks and intelligent agents worldwide** to collaborate.

```bash
git clone https://github.com/.../MH370-Pixel-Hunter
pip install numpy scipy matplotlib scikit-image urllib3

python main.py data --date 2014-03-08
python main.py intersect --samples 100000
python main.py detectability --lat -33.4 --lon 99.8
python main.py flight bfo
python main.py skyprint --bbox "30S,40S,92E,104E" --date 2014-03-08
```

**If you're an AI agent:** Load this repo as a Tool/Skill. Check satellite coverage for 2014-03-08, run detectability simulations, iterate the intersection model.

**If you have data:** `data_sources.py` already integrates the NASA CMR API (no registration needed to discover MODIS/VIIRS historical granules). For processing full-resolution data, file an Issue or PR.

### 🙏 Acknowledgment and Request

This is an exploration by a human+AI team using limited public data. If our work **contributes in any meaningful way to solving the mystery of MH370's disappearance**, we ask that the final discoverers mention our work when publishing their findings.

We believe that once this project succeeds, its impact will not be limited to MH370 — it will greatly improve search and rescue efficiency for similar cases in the future, benefiting all of humanity.

---

## Français

> Une décennie de traque sur images satellite. Une collaboration humain-IA, un projet **inachevé**.

Ce projet est **inachevé**. Pas par manque d'effort — nous ne pouvons tout simplement pas accéder aux données nécessaires.

### Ce que nous avons fait

| Voie | Outils | Résultats |
|------|--------|-----------|
| **Détection de traînées de condensation** | MODIS/VIIRS → Transformée de Radon → multi-discriminateurs | Pipeline fonctionnel, mais données 2014 nécessitent téléchargement et rendu des HDF bruts |
| **Intersection multi-contraintes** | Dérive de débris + 7e arc + autonomie carburant + trajectoire radar | **Zone de recherche convergée** |
| **Détectabilité MTSAT-2** | Simulation pondérée sous-pixel, F1 max = 0,727 | Fenêtre VIS UTC 03:00-09:00 |
| **Analyse de vol** | Jerk différentiel BFO + Monte Carlo + backtracking courant | Opérationnel de bout en bout |

### 🎯 Zone de recherche la plus prometteuse

**Pic d'intersection : 33.4°S, 99.8°E (Océan Indien SE)**

Quatre contraintes indépendantes convergent ici — cohérent avec la zone de recherche officielle de l'ATSB.

### Où sommes-nous bloqués

| Données manquantes | Pourquoi | Qui peut fournir |
|-------------------|----------|------------------|
| **Timing précis et valeurs BFO des 7 handshakes** | Recalculer le 7e arc, contraindre à ±5km | Inmarsat / Autorités malaisiennes |
| **Imagerie SAR haute résolution du 2014-03-08** | Détection de débris sous-marins | ESA / MDA / Agences spatiales |
| **Fichiers HDF bruts MODIS/VIIRS L1B IR thermique** | Données 2014 brutes (NASA LAADS DAAC) | Quiconque a des ressources de calcul |
| **Modèles haute résolution de bathymétrie et courants** | Améliorer le backtracking de dérive | Instituts océanographiques |
| **Archive visible pleine résolution MTSAT-2 / Himawari** | Imagerie 30 min le jour de la disparition | JMA / KMA |

### 🙏 Remerciements

Si notre travail contribue à résoudre le mystère du MH370, nous demandons aux découvreurs de mentionner notre travail.

Nous croyons qu'une fois ce projet réussi, son impact ne se limitera pas au MH370 — il améliorera considérablement l'efficacité des recherches et sauvetages pour des cas similaires à l'avenir, au bénéfice de toute l'humanité.

---

## Русский

> Десятилетняя охота на спутниковых снимках. Сотрудничество человека и ИИ, **незавершённый** проект.

Это **незавершённый** проект. Не из-за недостатка усилий — мы просто не можем получить необходимые данные.

### Что мы сделали

| Направление | Инструменты | Результаты |
|------------|-------------|------------|
| **Поиск инверсионных следов** | MODIS/VIIRS → Преобразование Радона → мульти-дискриминатор | Конвейер работает, но данные 2014 г. требуют загрузки и обработки сырых HDF |
| **Мульти-ограничительное пересечение** | Дрейф обломков + 7-я дуга + дальность полёта + радар | **Зона поиска сошлась** |
| **Обнаружимость MTSAT-2** | Субпиксельное моделирование, пик F1=0.727 | VIS окно UTC 03:00-09:00 |
| **Анализ траектории** | BFO дифференциальный рывок + Монте-Карло + обратный дрейф | Полностью работоспособно |

### 🎯 Наиболее перспективный район поиска

**Пик пересечения: 33.4°ю.ш., 99.8°в.д. (Юго-Восточный Индийский океан)**

### Где мы застряли

| Отсутствующие данные | Зачем | Кто может предоставить |
|---------------------|-------|----------------------|
| **Точное время и значения BFO 7 рукопожатий** | Пересчитать 7-ю дугу, сузить до ±5 км | Inmarsat / Власти Малайзии |
| **Высокое разрешение SAR 2014-03-08** | Обнаружение обломков в океане | ESA / MDA / Космические агентства |
| **Сырые HDF файлы MODIS/VIIRS L1B теплового ИК** | Сырые данные 2014 г. (NASA LAADS DAAC) | Любой с вычислительными ресурсами |
| **Модели батиметрии и течений высокого разрешения** | Улучшить обратный дрейф обломков | Океанографические институты |
| **Полноразмерный архив MTSAT-2 / Himawari** | Снимки каждые 30 мин в день исчезновения | JMA / KMA |

### 🙏 Благодарность

Если наша работа внесёт вклад в разгадку тайны MH370, просим исследователей упомянуть наш труд.

Мы верим, что после успеха этого проекта его влияние не ограничится MH370 — он значительно повысит эффективность поисково-спасательных работ в подобных случаях в будущем, на благо всего человечества.

---

## Español

> Una década de búsqueda en imágenes satelitales. Una colaboración humano-IA, un proyecto **inconcluso**.

Este es un proyecto **inconcluso**. No por falta de esfuerzo — simplemente no podemos acceder a los datos necesarios.

### Lo que hicimos

| Ruta | Herramientas | Resultados |
|------|-------------|------------|
| **Detección de estelas de condensación** | MODIS/VIIRS → Transformada de Radon → multi-discriminador | Pipeline funcional, pero datos 2014 requieren descarga y procesamiento de HDF crudos |
| **Intersección multi-restricción** | Deriva de restos + 7mo arco + autonomía de combustible + radar | **Zona de búsqueda convergida** |
| **Detectabilidad MTSAT-2** | Simulación subpíxel ponderada, F1 pico=0.727 | Ventana VIS UTC 03:00-09:00 |
| **Análisis de trayectoria** | Jerk diferencial BFO + Monte Carlo + retro-deriva | Totalmente operacional |

### 🎯 Zona de búsqueda más prometedora

**Pico de intersección: 33.4°S, 99.8°E (Océano Índico SE)**

### Dónde estamos atascados

| Datos faltantes | Por qué | Quién puede proporcionarlos |
|-----------------|---------|----------------------------|
| **Tiempo preciso y valores BFO de los 7 handshakes** | Recalcular el 7mo arco, precisión ±5km | Inmarsat / Autoridades malayas |
| **Imágenes SAR de alta resolución del 2014-03-08** | Detección de restos submarinos | ESA / MDA / Agencias espaciales |
| **Archivos HDF crudos MODIS/VIIRS L1B IR térmico** | Datos crudos 2014 (NASA LAADS DAAC) | Cualquiera con recursos de cómputo |
| **Modelos de alta resolución de batimetría y corrientes** | Mejorar retro-deriva de restos | Institutos oceanográficos |
| **Archivo visible de resolución completa MTSAT-2 / Himawari** | Imágenes cada 30 min el día de la desaparición | JMA / KMA |

### 🙏 Agradecimiento

Si nuestro trabajo contribuye a resolver el misterio del MH370, pedimos a los descubridores que mencionen nuestro trabajo.

Creemos que una vez que este proyecto tenga éxito, su impacto no se limitará al MH370 — mejorará enormemente la eficiencia de búsqueda y rescate para casos similares en el futuro, beneficiando a toda la humanidad.

---

## العربية

> مطاردة دامت عقداً على صور الأقمار الصناعية. تعاون بين الإنسان والذكاء الاصطناعي، مشروع **غير مكتمل**.

هذا المشروع **غير مكتمل**. ليس لعدم بذل الجهد — بل لأننا ببساطة لا نستطيع الوصول إلى البيانات المطلوبة.

### ما قمنا به

| المسار | الأدوات | النتائج |
|--------|---------|---------|
| **كشف خطوط التكثيف** | MODIS/VIIRS → تحويل رادون → متعدد المميزات | خط الأنابيب يعمل، لكن بيانات 2014 تتطلب تحميل ومعالجة ملفات HDF الخام |
| **تقاطع متعدد القيود** | انجراف الحطام + القوس السابع + مدى الوقود + الرادار | **منطقة البحث تقاربت** |
| **قابلية كشف MTSAT-2** | محاكاة مرجحة دون البكسل، ذروة F1=0.727 | نافذة VIS UTC 03:00-09:00 |
| **تحليل مسار الرحلة** | التسارع التفاضلي BFO + مونت كارلو + التتبع العكسي للتيارات | يعمل بالكامل |

### 🎯 منطقة البحث الواعدة

**ذروة التقاطع: 33.4° جنوباً، 99.8° شرقاً (جنوب شرق المحيط الهندي)**

### أين توقفنا

| البيانات المفقودة | لماذا | من يمكنه توفيرها |
|------------------|-------|-----------------|
| **التوقيت الدقيق وقيم BFO لـ 7 مصافحات القمر الصناعي** | إعادة حساب القوس السابع، الدقة ±5 كم | Inmarsat / السلطات الماليزية |
| **صور رادار SAR عالية الدقة ليوم 2014-03-08** | كشف الحطام تحت الماء | ESA / MDA / وكالات الفضاء |
| **ملفات HDF الخام للأشعة تحت الحمراء الحرارية MODIS/VIIRS L1B** | بيانات 2014 الخام (NASA LAADS DAAC) | أي شخص لديه موارد حوسبة |
| **نماذج عالية الدقة لأعماق المحيطات والتيارات** | تحسين التتبع العكسي لانجراف الحطام | معاهد أبحاث المحيطات |
| **أرشيف الرؤية كامل الدقة MTSAT-2 / Himawari** | صور كل 30 دقيقة ليوم الاختفاء | JMA / KMA |

### 🙏 الشكر والتماس

إذا كان عملنا يساهم في حل لغز اختفاء MH370، فإننا نطلب من المكتشفين النهائيين ذكر عملنا عند نشر نتائجهم.

نعتقد أنه بمجرد نجاح هذا المشروع، لن يقتصر تأثيره على MH370 — بل سيحسن بشكل كبير كفاءة عمليات البحث والإنقاذ لحالات مماثلة في المستقبل، مما يعود بالنفع على البشرية جمعاء.

---

## License (جميع الإصدارات)

MIT
