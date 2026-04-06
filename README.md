# 创伤可视化系统 — 预测服务

同济医院创伤急救数据可视化平台的 **时空因导预测模块**，独立 Python 微服务。

基于 2151 条真实急救抢救室病历数据，预测上海市各区域/时段/季节的创伤伤因概率分布。

---

## 功能

- **T1** 某地区 + 时段 + 季节 → 伤因概率分布（综合预测）
- **T2** 某伤因 → 各时段/季节历史分布
- **T3** 某伤因 → 各上海区域发生分布（热力图数据）
- **T4** 某区域 → 伤因构成比（地区画像）
- **T5** 某时段 → 伤因分布
- **T6** 某时段 + 某伤因 → 区域空间分布

所有预测结果附带 **置信度**（high/medium/low）和 **样本量**，样本不足时自动降级兜底。

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 主语言 |
| FastAPI | 0.111 | Web 框架 |
| uvicorn | 0.30 | ASGI 服务器 |
| joblib | 1.4 | 模型序列化 |
| SQLAlchemy + PyMySQL | 2.0 / 1.1 | MySQL 连接（全量训练时使用） |
| pandas / numpy | 2.2 / 1.26 | 数据处理 |

---

## 项目结构

```
prediction_service/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── config.py                  # 配置（DB、模型路径、端口）
│   ├── models/
│   │   ├── statistical_model.py   # 统计层：条件概率模型
│   │   └── feature_builder.py     # 特征配置表，新增字段只改这里
│   ├── services/
│   │   ├── data_service.py        # 从 MySQL 读训练数据（全量训练时使用）
│   │   └── training_service.py    # 训练 + 增量更新
│   ├── api/
│   │   └── prediction.py          # API 路由
│   └── utils/
│       └── geo_utils.py           # 上海地址 → 区名解析（覆盖231条地址规则）
├── models/                        # 持久化模型文件（.pkl + .meta.json）
├── train_offline.py               # 离线训练脚本（从 Excel 训练，不依赖数据库）
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（复制示例文件）
cp .env.example .env
# 编辑 .env：填写 MySQL 连接串

# 4. 启动服务
uvicorn app.main:app --reload --port 8000

# 5. 查看 API 文档
open http://localhost:8000/docs
```

### 首次训练模型

**方式一：从数据库全量训练（推荐，部署后使用）**

```bash
# 服务启动后，在 Swagger Docs 页面调用：
# POST /api/model/train
# 或通过 curl：
curl -X POST http://localhost:8000/api/model/train
```

**方式二：从 Excel 离线训练（无数据库时使用）**

```bash
python train_offline.py
```

---

## API 接口

### 预测接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/predict/comprehensive` | T1：综合预测（区域+时段+季节，均可不传表示全部） |
| GET | `/predict/cause-by-period` | T2：时段（+可选季节）→ 伤因分布 |
| GET | `/predict/cause-by-season` | T2：季节 → 伤因分布 |
| GET | `/predict/cause-by-district` | T4：区域 → 伤因分布 |
| GET | `/predict/time-distribution` | T2：伤因 → 时段/季节分布 |
| GET | `/predict/district-distribution` | T3/T6：各区域发生量（热力图数据） |
| GET | `/predict/district-profile` | T4：地区画像（时段/季节/伤因分布，不传则全市） |
| GET | `/predict/district-by-period-cause` | T6：时段+伤因 → 区域分布 |

### 模型管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/model/status` | 当前模型版本、样本量、评估指标 |
| GET | `/api/model/version` | 版本详情（base/incremental/total） |
| POST | `/api/model/train` | 全量训练（从数据库读取所有记录） |
| POST | `/api/model/incremental` | **增量训练（Java推送新数据，不依赖数据库）** |
| POST | `/api/model/trigger-update` | 增量更新（旧接口，需Python可达数据库） |
| GET | `/api/model/history` | 历史模型版本列表 |

---

## 与主系统集成

```
[前端 Vue2 :8080] → [Java SpringBoot :9090] → [本服务 :8000]
                                                      ↕
                                                  [MySQL]（仅全量训练时）
                                                  [models/*.pkl]
```

Java 后端通过 `PredictionController.java` 代理转发请求，前端无需直接访问本服务。

---

## 模型更新机制

### 全量训练（首次初始化）

Java 前端点击「全量训练」→ Java 调用 Python `POST /api/model/train` → Python 从数据库读取所有 `injuryrecord` 记录训练模型。

### 增量训练（Java 推数据模式）

用户导入新 Excel → Java 解析入库 → 新增记录成功后，Java **异步**将新增的 `InjuryRecord` 序列化为 JSON，推送给 Python `POST /api/model/incremental`。

**特点：**
- Python 不连接数据库，全程由 Java 推数据
- 非阻塞：预测服务不可达时只打 warn，不影响导入主流程
- 从现有模型计数表还原历史样本 + 合并新数据后重拟合
- 每次增量更新生成带时间戳版本文件（`v{n}_{datetime}_incr`），可追溯

### 评估方式

采用**时间序列切分**（前80%训练，后20%测试），out-of-sample 评估，避免训练集泄露导致准确率虚高。当前基线准确率约 **42%**（5分类，随机基线 20%）。

---

## 参数说明

### 时间段（time_period）

| 值 | 名称 | 时间范围 |
|----|------|---------|
| 0 | 夜间 | 00:00–07:59 |
| 1 | 早高峰 | 08:00–09:59 |
| 2 | 午高峰 | 10:00–11:59 |
| 3 | 下午 | 12:00–16:59 |
| 4 | 晚高峰 | 17:00–19:59 |
| 5 | 晚上 | 20:00–23:59 |

### 季节（season）

| 值 | 名称 | 月份 |
|----|------|------|
| 0 | 春季 | 3–5月 |
| 1 | 夏季 | 6–9月 |
| 2 | 秋季 | 10–12月 |
| 3 | 冬季 | 1–2月 |

### 伤因（injury_cause）

| 值 | 名称 |
|----|------|
| 0 | 交通伤 |
| 1 | 高坠伤 |
| 2 | 机械伤 |
| 3 | 跌倒 |
| 4 | 其他 |

---

## 关联项目

- 前端：[tongji-hospital-front](https://github.com/hfutwz/tongji-hospital-front)
- 后端：[tongji-hospital-back](https://github.com/hfutwz/tongji-hospital-back)
