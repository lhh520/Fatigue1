import datetime

from public_mysql import connect   #调用public_mysql文件的connect函数
import cv2
class MySQL():
    def implement(self):
        '''执行SQL语句'''
        db = connect()
        cursor = db.cursor()
        for i in range(1):
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("insert into user(eye,mou,rate,reason,update_time) \
                                  values('%d','%d','%d','%s','%s')" % \
                           (user.getAge(), user.getName(), dt, dt))
            sql = """SELECT * FROM `user`"""
            try:
                cursor.execute(sql)
                result = cursor.fetchone()
                db.commit()
                print('查询结果：', result)
            except Exception:
                db.rollback()
                print("查询失败")

        cursor.close()
        db.close()
def implement1(eye,mou,rate,pico,reason):
    '''执行SQL语句'''
    db = connect()
    cursor = db.cursor()
    for i in range(1):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        try:
            cursor.execute("insert into user(eye,mou,rate,pico,reason,update_time) \
                                          values('%d','%d','%d','%s','%s','%s')" % \
                           (eye, mou, rate, pico,reason, dt))
            result = cursor.fetchone()
            db.commit()
            print('查询结果：', result)
        except Exception:
            db.rollback()
            print("查询失败")

    cursor.close()
    db.close()


if __name__ == '__main__':
    img=cv2.imread("/Fatigue\\images\\pipeline.png")
    dt = datetime.datetime.now().strftime("%H%M%S")
    print(type(dt))
    cv2.imwrite('E:\\images\\'+dt+'.jpg',img)


