#-*-coding:utf-8-*-
import mysql.connector as mc
from mysql.connector import errorcode
import json
import numpy as np
import copy

'''database'''
config = {
  'user': '',
  'password': '',
  'host': '',
  'database': '',
  'raise_on_warnings': True,
}

class engine:
    def __init__(self,n_SAM,n_RADAR,name,snum,Number = 0):
        self.n_SAM = n_SAM
        self.n_RADAR = n_RADAR
        self.Number = Number
        self.name = name
        self.snum = str(snum)
    def connect(self):
        '''Connect database'''
        try:
            self.cnx = mc.connect(**config)
        except mc.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Wrong user name and password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("The database does not exist")
            else:
                print(err)
        else:
            print('Database {} is connected'.format(config['database']))
            self.cursor = self.cnx.cursor()
    def connect_data(self):
        '''Calculate the probability of being destroyed'''
        p_destory = []
        p_real = []
        live = []
        sql = "SELECT Poss_destroy,Activity FROM %s WHERE Type = 'UAV' and step != 0  and mod(step, %s) = 0 " % (self.name,self.snum)
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            for row in results:
                judge = 0
                lo = json.loads(row[0])
                if type(lo) == int:
                    p_destory.append(float(lo))
                else:
                    if len(lo) == 1:
                        p_destory.append(float(lo[0]))
                    else:
                        a = 1
                        for h in range(judge):
                            a *= (1 - lo[h])
                            b = 1 - lo[h]
                            p_destory.append(b)
                live.append(row[1] / 1000)
                p_real.append(1 - (row[1] / 1000))
        except:
            print("Error: unable to fecth data")
        # 关闭数据库连接
        self.cnx.close()
        return p_destory,p_real
    def get_data(self):
        '''Get training data'''
        sql = "SELECT X,Y ,ATT_radius,Attribute FROM %s WHERE step != 0 and mod(step, %s) = 0" % (self.name,self.snum)
        uav_position = []
        radar_position = []
        sam_position = []
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        tp = 1
        for row in results:
            if tp == 1:
                uav_position.append([row[0],row[1]])
                temp1 = copy.deepcopy(row[0])
                temp2 = copy.deepcopy(row[1])
                tp += 1
            elif 1<tp<=(self.n_SAM+1) :
                sam_position.append([temp1,temp2,row[0], row[1],row[2], row[3]])
                tp += 1
            elif tp == (1+self.n_RADAR+self.n_SAM):
                radar_position.append([temp1,temp2,row[0], row[1],row[2], row[3]])
                tp = 1
            else:
                radar_position.append([temp1,temp2,row[0], row[1],row[2], row[3]])
                tp += 1
        X1 = np.array(radar_position).reshape(-1,self.n_RADAR,6)
        X2 = np.array(sam_position).reshape( -1,self.n_SAM, 6)
        return X1,X2
    def get_label_real(self):
        '''Get label data'''
        _,p_real = self.connect_data()
        return p_real



