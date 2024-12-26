import datetime as t
import lotto

# 고객의 성별 정보 저장
def choice_gender(guest_log):
    print("=" * 20)
    gender = input("\n성별입력\n1. 남자 / 2. 여자")
    print("=" * 20, end="\n")

    if gender == '1':
        guest_log["gender"] = 'man'

    elif gender == '2':
        guest_log["gender"] = 'woman'

    else:
        return True

    return False

# 고객의 나이대 정보 저장
def choice_age(guest_log):
    print("=" * 20)
    print("\n나이 입력\n0. 10대 미만\n1. 10대\n2. 20대\n3. 30대\n4. 40대\n5. 50대\n6. 60대\n7. 60대 이상")
    print("=" * 20, end="\n")
    age = input("나이를 선택해주세요.")

    if age == '0':
        guest_log['age'] = 0

    elif age == '1':
        guest_log['age'] = 10

    elif age == '2':
        guest_log['age'] = 20

    elif age == '3':
        guest_log['age'] = 30

    elif age == '4':
        guest_log['age'] = 40

    elif age == '5':
        guest_log['age'] = 50

    elif age == '6':
        guest_log['age'] = 60

    elif age == '7':
        guest_log['age'] = 70

    else:
        return True

    return False

def select_goods(tmp, goods):
    tmp_list = list(goods.keys())
    print("=" * 20)

    for i in tmp_list:
        print("\n{}.\t{}\t/\t금액 : {}\t/\t재고 : {}".format(i, goods[i]["품목"], goods[i]["가격"], goods[i]["재고"]))
    print("=" * 20, end="\n")

    s_num = 0
    while True:
        s_num = input("\n구매 상품 번호 : ")
        if s_num in goods:
            break

    while True:
        try:
            count = int(input("\n구매 상품 수량 : "))
            if count > goods[s_num-1]:
                print("재고가 부족합니다. 다시 입력해주세요.")
                continue
            else:
                break
        except:
            continue

    if s_num == '10':
        lotto.main(count)

    while True:
        print("=" * 20)
        print("제품 : {} / 수량 : {}".format(goods[s_num]['품목'], count))
        print("=" * 20, end="\n")
        num = input("1. 확인 / 2. 취소")
        print("=" * 20, end="\n")
        if num == '1':
            tmp[s_num] = count
            return False
        elif num == '2':
            return True

def flow_choice(guest_log, goods):
    boolean = True
    service = 0
    tmp_goods = guest_log["판매"]

    while boolean:
        boolean = select_goods(tmp_goods, goods)

    guest_log["판매"] = tmp_goods

    while True:
        print("=" * 20)
        end_flag = input("1. 다음 / 2. 물품 추가선택")
        print("=" * 20, end="\n")

        if end_flag == '1':
            return False

        elif end_flag == '2':
            return True

def calc_cost(tmp_dic, goods):
    total = 0
    tmp_list = list(tmp_dic.keys())
    sale_dic = {}
    count = 0
    for i in tmp_list:
        tmp = {}
        count += 1
        t = goods[i]["가격"] * tmp_dic[i]
        total += t
        tmp["품목"] = goods[i]["품목"]
        tmp["수량"] = tmp_dic[i]
        tmp["총금액"] = t
        sale_dic[count] = tmp

    return sale_dic, total

def select_payment():
    print("=" * 20)
    payment = input("1. 카드 / 2. 현금")
    print("=" * 20, end="\n")
    tmp = 0

    if payment == '1':
        tmp = 1

    elif payment == '2':
        tmp = 2

    else:
        return True, tmp

    return False, tmp

def select_card(sale_dic, total, guest_log):
    print("=" * 5, "영수증", "=" * 5)
    print("품목\t수량\t금액")

    for i in sale_dic.keys():
        print("{}\t{}\t{}".format(sale_dic[i]["품목"], sale_dic[i]["수량"], sale_dic[i]["총금액"]))

    print("=" * 20, end="\n")
    print("\ntotal : {}".format(total))
    print("=" * 20, end="\n")
    tmp = input("1. 확인 / 2. 취소")
    print("=" * 20, end="\n")

    if tmp == '1':
        print("결제완료")
        print("=" * 10)
        guest_log["결제"] = "card"
        guest_log["판매금액"] = total
        guest_log["거스름돈"] = None
        return False

    elif tmp == '2':
        print("결제취소")
        print("=" * 10)
        return True

    else:
        print("다시 입력해주세요.")
        return True

def select_cash(sale_dic, total, guest_log):
    print("=" * 5, "영수증", "=" * 5)
    print("품목\t수량\t금액")

    for i in sale_dic.keys():
        print("{}\t{}\t{}".format(sale_dic[i]["품목"], sale_dic[i]["수량"], sale_dic[i]["총금액"]))

    print("=" * 20, end="\n")
    print("\ntotal : {}".format(total))
    print("=" * 20, end="\n")

    while True:
        tmp = input("\n받은 현금 : ")

        try:
            tmp = int(tmp)
            break
        except:
            continue

    if tmp - total < 0:
        print("=" * 20, end="\n")
        print("받은 현금이 부족합니다.")
        print("=" * 20, end="\n")
        return True

    else:
        print("=" * 20, end="\n")
        print("\n결제완료")
        print("=" * 20, end="\n")
        print("잔돈 : {}원".format(tmp - total))
        print("=" * 20, end="\n")
        guest_log["결제"] = "cash"
        guest_log["판매금액"] = total
        guest_log["거스름돈"] = (tmp - total)

def flow_payment(guest_log, goods):
    boolean = True
    select_num = 0
    tmp_goods = guest_log["판매"]
    sale_dic, total = calc_cost(tmp_goods, goods)

    while boolean:
        boolean, select_num = select_payment()

    boolean = True
    if select_num == 1:
        while boolean:
            boolean = select_card(sale_dic, total, guest_log)

    elif select_num == 2:
        while boolean:
            boolean = select_cash(sale_dic, total, guest_log)

    return False

def main(goods):
    now = t.datetime.now()
    guest_log = {"판매" : {}}
    print("=" * 20)
    print(now)
    print("=" * 20, end="\n")
    while choice_gender(guest_log):
        print("\n다시 입력해주세요.")

    while choice_age(guest_log):
        print("\n다시 입력해주세요.")

    while flow_choice(guest_log, goods):
        continue

    while flow_payment(guest_log, goods):
        continue

    now_time = "{}-{}-{} / {}:{}".format(now.year, now.month, now.day, now.hour, now.minute)

    guest_log["일시"] = now_time

    print("=" * 20, end="\n")

    return(guest_log)