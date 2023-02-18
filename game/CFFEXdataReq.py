import requests

# 'http://www.cffex.com.cn/sj/historysj/201504/zip/201504.zip'
baseUrl = 'http://www.cffex.com.cn/sj/historysj/%s/zip/%s.zip'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        if len(response.history) > 0 :
            print(filename + '下载失败')
            return
        # 以二进制格式打开一个文件只用于写入，如果有同名filename则会覆盖重写
        with open('CFFEX/' + filename, 'wb') as f:
            f.write(response.content)
        print(filename + '下载成功')
    else:
        print(filename + '下载失败')


for index in range(15, 24):
    for m in range(len(months)):
        year = '20' + str(index)
        month = months[m]
        tag = year + month
        url = baseUrl % (tag, tag)
        print(url)
        download(url, tag + '.zip')
