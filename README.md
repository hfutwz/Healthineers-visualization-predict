# 创伤可视化系统 — 预测服务

同济医院创伤急救数据可视化平台的 **时空因导预测模块**，独立 Python 微服务。

基于 2151 条真实急救抢救室病历数据，预测上海市各区域/时段/季节的创伤伤因概率分布。

---

## 功能

- **T1** 某地区 + 时段 + 季节 → 伤因概率分布
- **T2** 某伤因 → 各时段/季节历史分布
- **T3** 某伤因 → 各上海区域发生分布（热力图数据）
- **T4** 某区域 → 伤因构成比
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
| scikit-learn | 1.5 | 机器学习（Phase 2） |
| joblib | 1.4 | 模型序列化 |
| SQLAlchemy + PyMySQL | 2.0 / 1.1 | MySQL 连接 |
| pandas / numpy | 2.2 / 1.26 | 数据处理 |

---

## 项目结构

```
prediction_service/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── config.py                  # 配置（DB、模型路径、端口）
│   ├── models/
│   │   ├── statistical_model.py   # 统计层：条件概率模型（MVP）
│   │   ├── ml_model.py            # ML层：多项式逻辑回归（Phase 2）
│   │   └── feature_builder.py     # ★ 特征配置表，新增字段只改这里
│   ├── services/
│   │   ├── data_service.py        # 从 MySQL 读训练数据
│   │   └── training_service.py    # 训练 + 增量更新
│   ├── api/
│   │   └── prediction.py          # API 路由（8个接口）
│   └── utils/
│       └── geo_utils.py           # 上海地址 → 区名解析
├── models/                        # 持久化模型文件（.pkl + .meta.json）
├── tests/
│   └── test_prediction.py
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

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/predict/cause-by-period` | 时段（+可选季节）→ 伤因分布 |
| GET | `/predict/cause-by-district` | 区域 → 伤因分布 |
| POST | `/predict/comprehensive` | 区域 + 时段 + 季节 → 综合预测 |
| GET | `/predict/time-distribution` | 伤因 → 时段/季节分布 |
| GET | `/predict/district-distribution` | 各区域发生量（热力图数据） |
| GET | `/api/model/status` | 当前模型版本、样本量、评估指标 |
| POST | `/api/model/trigger-update` | 触发增量更新（导入新数据后调用） |
| GET | `/api/model/history` | 历史模型版本列表 |

---

## 与主系统集成

```
[前端 Vue2 :8080] → [Java SpringBoot :9090] → [本服务 :8000]
                                                      ↕
                                                  [MySQL]
                                                  [models/*.pkl]
```

Java 后端通过 `PredictionController.java` 代理转发请求，前端无需直接访问本服务。

---

## 模型更新机制

- **自动触发**：用户导入新 Excel 后，Java 后端异步调用 `/api/model/trigger-update`
- **增量条件**：新增数据 ≥ 30 条才触发重训（避免频繁更新）
- **版本管理**：每次更新保存带时间戳的版本文件，可随时回滚
- **特征扩展**：修改 `feature_builder.py` 中的 `FEATURE_CONFIG` 即可新增特征，无需改其他代码

---

## 关联项目

- 前端：[tongji-hospital-front](https://github.com/hfutwz/tongji-hospital-front)
- 后端：[tongji-hospital-back](https://github.com/hfutwz/tongji-hospital-back)
