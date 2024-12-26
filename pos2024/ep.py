import management
import datetime as t

def re_stock(goods):
    f = open("재고/goods.txt", "w")

    for i in goods.keys():
        f.write("{}/{}/{}/{}/{}\n".format(i, goods[i]["분류"], goods[i]["품목"], goods[i]["가격"], goods[i]["재고"]))

    f.close()

def make_day_sale(month, day, day_sale):
    f = open("관리/" + month + day + ".txt", "w")

    for i in day_sale.keys():
        f.write("{}/{}\n".format(i, day_sale[i]))

    f.close()

def make_month_sale(month, day_sale):
    tmp_dic = management.month_margin(month)
    month_sale = tmp_dic[month]
    f_dic = {}

    for i in month_sale.keys():
        total = month_sale[i] + day_sale[i]
        f_dic[i] = total

    f = open("관리/" + month + "-total.txt", "w")

    for i in f_dic.keys():
        f.write("{}/{}\n".format(i, f_dic[i]))

    f.close()

def main(goods, day_sale):
    now = t.datetime.now()
    month = now.month
    day = now.day

    if month < 10:
        month = '0' + str(month)

    else:
        month = str(month)

    if day < 10:
        day = '0' + str(day)

    else:
        day = str(day)

    re_stock(goods)
    make_day_sale(month, day, day_sale)
    make_month_sale(month, day_sale)