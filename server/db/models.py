from datetime import datetime
from configs.settings import settings
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Enum, Float

Base = declarative_base()

# ---------------------------------------------------------
# 业务数据表
# ---------------------------------------------------------
class EmeiScenicSpot(Base):
    __tablename__ = 'scenic_spots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="景点名称")
    altitude = Column(Float, comment="海拔(米)")
    location_type = Column(String(50), comment="地点类型")
    description = Column(Text, comment="详细描述")

# ---------------------------------------------------------
# 知识库文档户口本
# ---------------------------------------------------------
class KnowledgeFile(Base):
    __tablename__ = 'knowledge_files'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 核心身份
    file_name = Column(String(255), nullable=False, comment="文件名")
    file_path = Column(String(512), nullable=False, comment="文件存储路径")

    # 查重：文件的数字指纹
    file_hash = Column(String(64), unique=True, nullable=False, comment="文件内容的哈希值")

    # 状态流转: PENDING(待处理) -> PROCESSING(加工中) -> SUCCESS(成功) / FAILED(失败)
    status = Column(String(20), default="PENDING", comment="处理状态")

    # 统计信息
    node_count = Column(Integer, default=0, comment="切分出的节点数量")
    error_msg = Column(Text, nullable=True, comment="如果失败，记录报错信息")

    created_at = Column(DateTime, default=datetime.now, comment="首次上传时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="最后更新时间")

# ---------------------------------------------------------
# 数据库初始化逻辑
# ---------------------------------------------------------
engine = create_engine(settings.mysql_url)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("MySQL 表结构 'scenic_spots' 已自动创建！")

if __name__ == "__main__":
    init_db()