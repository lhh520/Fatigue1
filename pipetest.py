from multiprocessing import Process, Pipe
import time


def consumer(p, name):
    left, right = p
    left.close()
    while True:
        try:
            baozi = right.recv()
            file = open('/Fatigue\\mq.txt', 'w')

            for i in range(1, 2):
                file.write((str(baozi)) + '\n')  # 身份证号
            file.close()
            print('%s 收到包子:%s' % (name, baozi))
        except EOFError:
            right.close()
            break


def producer(seq, p):
    left, right = p
    right.close()
    for i in seq:
        file = open('/Fatigue\\mq.txt', encoding ="utf-8")
        stt=file.read()
        print(type(stt))
        print(int(stt))
        #print(file.read())
        file.close()
        left.send(i)
        time.sleep(1)
    else:
        left.close()
def divide(a,b):
    return a/b

if __name__ == '__main__':
    left, right = Pipe()

    c1 = Process(target=consumer, args=((left, right), 'c1'))
    c1.start()

    seq = (i for i in range(10))
    producer(seq, (left, right))

    right.close()
    left.close()

    c1.join()
    print('进程间通信-管道-主进程')
    # try:
    #     divide(1, 0)
    # except:
    #     print("divide by 0")
    # else:
    #     print("the code is no problem")
    #
    # print("code after try catch,hello,world!")



