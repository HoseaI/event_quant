
import pymysql

database_ip = '192.168.239.130'
database_port = 3306
database_user = 'root'
database_password = '123456'


class mysqlManage(object):
    def __init__(self, host=database_ip, port=database_port, name=database_user, password=database_password):
        self.conn =  pymysql.connect(host=host, port=port, user=name,  password=password, database='quant', charset='utf8')
        self.cursor = self.conn.cursor()

    def execute(self, sql):
        self.cursor.execute(sql)
        ret = self.cursor.fetchall()
        self.conn.commit()
        return ret
    
    def close(self):
        self.cursor.close()
        self.conn.close()