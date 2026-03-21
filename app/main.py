"""
FastAPI 入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.prediction import router

app = FastAPI(
    title="创伤预测服务（时空因导算法）",
    description="基于上海同济医院急救数据的时空伤因分布预测 API",
    version="1.0.0",
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
    return {"service": "trauma-prediction", "docs": "/docs"}
