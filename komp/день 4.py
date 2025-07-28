import random
угадайка =random.randint(1,100)
попытки=75
for i in range(попытки):
 огурец=int(input("угадай"))
 if огурец==угадайка:
      print("Умничка)")
      exit()
 elif огурец>угадайка:
  print("Меньше")
 elif огурец<угадайка:
  print("Больше")
print("хаха")