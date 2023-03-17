import requests
import re
import json
import pandas as pd

# data is catched from eastmoney
url = 'http://13.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124029175978392917634_1679021002481&pn=1&pz=300&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs=m:124,m:125,m:305'

response = requests.get(url)

# 提取括号内容
content = str(response.content, 'UTF-8')
body = re.findall(r'[(](.*?)\);$', content)[0]

resp = json.loads(body)
data = resp['data']
indexes = data['diff']

codes = []
for i in indexes:
    codes.append(i['f12'])

output = pd.DataFrame({'code': codes})
output.to_csv('HKIndices.csv', index=False)

