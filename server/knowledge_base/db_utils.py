import hashlib
from server.db.models import SessionLocal, KnowledgeFile

# 用 Python 标准库 hashlib 里的 MD5 算法，
# 按照“分块读文件 → 不断喂给 md5 → 输出十六进制指纹”的套路算出一个文件的哈希（指纹）
def calculate_file_hash(file_path: str):
    """
    计算文件的 MD5 指纹，用于去重
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_or_create_file_record(file_path: str):
    """
    检查户口本：
    1. 如果文件hash已存在，返回 (existing_record, False) -> 也就是 skip
    2. 如果是新文件，创建一条 PENDING 记录，返回 (new_record, True) -> 也就是 process
    """
    session = SessionLocal()
    try:
        file_hash = calculate_file_hash(file_path)

        # 查重
        existing_file = session.query(KnowledgeFile).filter_by(file_hash=file_hash).first()
        if existing_file:
            if existing_file.status == "SUCCESS":
                return existing_file, False
            else:
                existing_file.status = "PENDING"
                existing_file.error_msg = None
                session.commit()
                return existing_file, True

        # 新户口登记
        import os
        new_file = KnowledgeFile(
            file_name=os.path.basename(file_path),
            file_path=str(file_path),
            file_hash=file_hash,
            status="PENDING"
        )
        session.add(new_file)
        session.commit()
        session.refresh(new_file)
        return new_file, True

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_file_status(file_id: int, status: str, node_count: int = 0, error: str = None):
    """更新处理结果"""
    session = SessionLocal()
    try:
        record = session.query(KnowledgeFile).filter_by(id=file_id).first()
        if record:
            record.status = status
            record.node_count = node_count
            if error:
                record.error_msg = str(error)[:1000]
            session.commit()
    finally:
        session.close()


