import csv
import pandas as pd
import xlrd

filepath1 = 'file/scopus2.csv'
list_scopus = []
list_scopus_title = []
with open(filepath1,'r',encoding='UTF-8') as f:
    reader = csv.reader(f)
    # header = next(reader)
    # print(header)

    for row in reader:
        #print(row[2])
        title = ''.join(filter(str.isalpha,row[2])).lower()  #只保留英文过滤
        list_scopus_title.append(title)
        list_scopus.append([row[12], row[3], row[0], row[2], row[16]])
print(len(list_scopus_title))
list_scopus_title = list_scopus_title[1:]
list_scopus = list_scopus[1:]
print("scopus论文数量 ",len(list_scopus_title))



filepath2 = 'file/webofscience.xls'
# df = pd.read_excel(filepath2,sheet_name='savedrecs')
# data = df.head()
# print(data)
# for p in data:
#     print(p)
list_web = []
list_web_title = []
data = xlrd.open_workbook(filepath2)
sh = data.sheet_by_name('savedrecs')
lent = sh.nrows
# print(sh.nrows)#有效数据行数

# da = sh.row_values(0)
# print(da)
# list_web.append([da[28],da[33],da[1],da[9],da[34]])

for i in range(1,lent):
    #print(i,"  ",sh.row_values(i))
    da = sh.row_values(i)
    title = ''.join(filter(str.isalpha,da[9])).lower()  #只保留英文过滤
    list_web_title.append(title)
    list_web.append([da[28],da[33],da[1],da[9],da[34]])


print("list_web_title",len(list_web_title),type(list_web_title))
print("web of science 论文数量",len(list_web))

filepath3 = 'file/gaze3.csv'

count2 = 0
with open(filepath3, 'a+', newline='',encoding='UTF-8') as f:
    writer = csv.writer(f)
    writer.writerow(['DOI', 'Publication Year','Authors', 'Article Title', 'Abstract'])
    for row in list_scopus:
        writer.writerow(row)
        count2 +=1

    for row2 in list_web:
        #print(row2,i)
        title = ''.join(filter(str.isalpha,row2[3])).lower()  #只保留英文过滤
        if title not in list_scopus_title:
            writer.writerow(row2)
            count2+=1

print('Scopus&WebOfScience total_num=',count2)

count = 0
for title in list_web_title:
    if title not in list_scopus_title:
        count += 1
print('重复论文 count = ',len(list_web_title)-count)

