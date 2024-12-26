def chk_goods(goods):
    goods_key = list(goods.keys())
    print("=" * 20)
    print("현재 재고 현황\n")
    print("=" * 20, end="\n")

    for i in goods_key:
        print(goods[i]['품목'], " : ", goods[i]["재고"])

    print("=" * 20, end="\n")

def chk_order(goods):
    goods_key = list(goods.keys())
    count = 0
    order_list = {}
    print("=" * 20)
    for i in goods_key:
        if goods[i]["재고"] < 20:
            print(goods[i]["품목"], '의 재고가', goods[i]['재고'], '개 입니다. 발주를 해주세요.')
            order_list[i] = goods[i]
            count += 1
    if count == 0:
        print("발주할 품목이 없습니다.\n")
        print("=" * 20, end="\n")

    else:
        print("=" * 20)
        select = input("1. 발주 / 2. 취소")
        print("=" * 20, end="\n")

        if select == '1':
            for i in list(order_list.keys()):
                if i not in list(goods.keys()):
                    continue
                goods[i]["재고"] = 150
            print("=" * 20)

        elif select == '2':
            print("취소되었습니다.")

        else:
            print("잘못 입력하였습니다. 다시 입력해주세요.")

        print("=" * 20, end="\n")

def main(goods):
    while True:
        print("=" * 20)
        select = input("1. 재고확인 / 2. 발주필요품목 / 5. 종료")
        print("=" * 20, end="\n")

        if select == '1':
            chk_goods(goods)

        elif select == '2':
            chk_order(goods)

        elif select == '5':
            break

        else:
            print("잘못 입력하였습니다. 다시 입력해주세요.")
            print("=" * 20, end="\n")