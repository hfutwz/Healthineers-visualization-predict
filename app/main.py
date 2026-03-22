"""
FastAPI 入口
启动时自动从 Excel 初始化基础模型（如尚未训练）
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.prediction import router, reload_model
from app.services.base_training_service import ensure_base_model

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时自动初始化模型"""
    logger.info("=== 创伤预测服务启动中 ===")

    # 1. 确保基础模型存在（首次启动从 Excel 训练）
    if ensure_base_model():
        logger.info("基础模型已就绪")
    else:
        logger.warning("基础模型不存在且 Excel 文件未找到，预测功能暂不可用，请调用 /api/model/train?source=excel 手动训练")

    # 2. 加载模型到内存
    reload_model()
    logger.info("模型已加载到内存，服务就绪")

    yield

    logger.info("=== 创伤预测服务关闭 ===")


app = FastAPI(
    title="创伤预测服务（时空因导算法）",
    description="基于上海同济医院急救数据的时空伤因分布预测 API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS（允许 Java 后端跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "trauma-prediction", "docs": "/docs", "version": "2.0.0"}


@app.get("/health", tags=["健康检查"])
def health():
    """健康检查接口"""
    from app.services import training_service
    from app.services.base_training_service import get_version_info
    status = training_service.get_model_status()
    version = get_version_info()
    return {
        "status": "ok" if status.get("model_ready") else "degraded",
        "model_ready": status.get("model_ready", False),
        "version": version.get("current_version", "未训练"),
        "mode": version.get("mode", "unknown"),
    }
