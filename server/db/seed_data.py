from server.db.models import SessionLocal, EmeiScenicSpot


def seed_emei_data():
    session = SessionLocal()
    spots = [
        EmeiScenicSpot(name="金顶", altitude=3079.3, location_type="自然景观",
                       description="峨眉山最高峰，有十方普贤像。"),
        EmeiScenicSpot(name="报国寺", altitude=551.0, location_type="寺庙", description="峨眉山第一座寺庙，门户所在。"),
        EmeiScenicSpot(name="万年寺", altitude=1020.0, location_type="寺庙",
                       description="供奉有宋代铸造的普贤骑象铜像。"),
        EmeiScenicSpot(name="清音阁", altitude=710.0, location_type="自然景观",
                       description="有'双桥清音'美誉，汇集黑白二水。")
    ]

    try:
        # 工业级习惯：先检查是否已经有数据，避免重复插入
        existing_count = session.query(EmeiScenicSpot).count()
        if existing_count == 0:
            session.add_all(spots)
            session.commit()
            print(f"成功灌入 {len(spots)} 条峨眉山种子数据！")
        else:
            print("数据库已有数据，跳过灌装。")
    except Exception as e:
        session.rollback()
        print(f"数据灌装失败: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    seed_emei_data()