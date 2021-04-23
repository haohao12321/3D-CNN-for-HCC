from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import xlrd

rbook = xlrd.open_workbook('C:/Users/cuihao/Desktop/train.xlsx')
# sheets方法返回对象列表,[<xlrd.sheet.Sheet object at 0x103f147f0>]
rbook.sheets()
# xls默认有3个工作簿,Sheet1,Sheet2,Sheet3
#rsheet = rbook.sheet_by_index(0)  # 取第一个工作簿
rsheet = rbook.sheet_by_index(1)
# 循环工作簿的所有行
fact = []
guess = []
for row in rsheet.get_rows():
    event = row[0]  # 品名所在的列
    eventv = event.value  # 项目名
    if eventv != 'event':  # 排除第一行
      fact.append(eventv)
      group = row[1]  # 价格所在的列
      groupv = group.value
      guess.append(groupv)
print(fact)
print(guess)

classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Output Class')
plt.ylabel('Target Class')
for first_index in range(len(confusion)):
  for second_index in range(len(confusion[first_index])):
   plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()