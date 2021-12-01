from xlwt import *


# 创建表格
file = Workbook(encoding='utf-8')
table = file.add_sheet('data')
table.write(0, 0, "向量1")
table.write(0, 1, "向量2")
table.write(0, 2, "向量3")
table.write(0, 3, "向量4")
table.write(0, 4, "向量5")
table.write(0, 6, "标签")

file.save('test.xlsx')