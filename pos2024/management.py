import datetime as t

# 월 매출을 파일에서 읽어와 딕셔너리 현태로 저장 후 리턴
def month_margin(month):
    f = open("관리/"+month+"-total.txt",'r')
    tmp_dic = {}
    tmp_month = {}

    while True:
        line = f.readline()
        if line == '':
            break
        line = line.rstrip('\n')
        tmp_list = line.split('/')
        tmp_month[tmp_list[0]] = int(tmp_list[1])

    tmp_dic[month] = tmp_month

    return tmp_dic

#
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

    while True:
        try:
            print("=" * 20)
            s_num = int(input("1. 일매출 / 2.월매출 / 5. 종료"))
            print("=" * 20, end="\n")

        except:
            continue

        if s_num == 1:
            print("="*5 + "일매출" + "="*5)
            for i in day_sale.keys():
                if i == 'card' or i == 'cash':
                    continue
                else:
                    print("{}. {} : {}".format(i, goods[i]['품목'], day_sale[i]))

            print("=" * 20, end="\n")
            print("{} : {}\n{} : {}\n". format('card', day_sale['card'], 'cash', day_sale['sale']))
            print("=" * 20, end="\n")

        elif s_num == 2:
            dic = month_margin(month)
            tmp_dic = dic[month]
            print("="*5 + "월매출" + "="*5)
            for i in tmp_dic.keys():
                if i == 'card' or i == 'cash':
                    continue
                else:
                    print("{}. {} : {}".format(i, goods[i]['품목'], tmp_dic[i]))

            print("=" * 20, end="\n")
            print("{} : {}\n{} : {}\n".format('card', tmp_dic['card'], 'cash', tmp_dic['cash']))
            print("=" * 20, end="\n")

        elif s_num == 5:
            break

        else:
            print("=" * 20)
            print("다시 입력해주세요.")
            print("=" * 20, end="\n")