import numpy as np
import pandas as pd

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

path = 'CFFEX/'



def getDelta(year, month, nextcode):
    y = '20' + year
    dir = path + y + month + '/'
    # 当月收盘
    current_month = 0
    for i in range(1, 31):
        if i < 10:
            day = '0' + str(i)
        else:
            day = str(i)
        print('当前日期' + y + month + day)
        file_path = dir + y + month + day + '_1.csv'
        if not os.path.exists(file_path):
            print(file_path + '文件不存在')
            continue
        df = pd.read_csv(file_path, encoding='gbk')
        # 查找当月
        cur_code = 'IC' + year + month
        # 找到对应合约代码的一行
        X = df.loc[df.合约代码.str.contains(cur_code, case=False)]
        if len(X) > 0:
            current_month = X['今收盘'].values
            print(cur_code + '存在，收盘价' + str(current_month))
        else:
            # 当月不存在，找下月
            next_code = 'IC' + nextcode
            X = df.loc[df.合约代码.str.contains(next_code, case=False)]
            if len(X) <= 0:
                print(file_path + '数据出错！！')
                return None
            next_month = X['今开盘'].values
            delta = current_month - next_month
            return [delta, current_month, next_month]
    return None

# 数据清洗，取当月最后一次的本月数据，和第一次的次月数据
deltas = []
codes = []
cur_m = []
next_m = []
for index in range(15, 24):
    for m in range(len(months)):
        year = str(index)
        month = months[m]
        if m < 11:
            nextcode = year + months[m + 1]
        else:
            # 下一年的code
            nextcode = str(index + 1) + months[0]
        result = getDelta(year, month, nextcode)
        if result is None:
            continue
        codes.append(year + month)
        deltas.append(result[0][0])
        cur_m.append(result[1][0])
        next_m.append(result[2][0])


output = pd.DataFrame({'time': codes, 'delta': deltas, 'currentMonth': cur_m, 'nextMonth': next_m})
output.loc[len(output.index)] = ['sum', sum(deltas), np.NAN, np.NAN]
output.to_csv(path + 'output.csv', index=False)
