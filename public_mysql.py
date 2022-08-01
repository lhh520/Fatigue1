import pymysql

def connect():
    '''连接MySQL数据库'''
    try:
        db = pymysql.connect(
            host='112.124.67.227',
            port=3306,
            user='lhh',
            passwd='123456789',
            db='facedetct',
            charset='utf8'
            )
        return db
    except Exception:
        raise Exception("数据库连接失败")
