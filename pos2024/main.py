import pay
import management
import gmanagement
import ep
import update

f = open("재고/goods.txt", "r")
goods = {}
day_sale = {"card" : 0, "cash" : 0}

while(True):
    tmp_dic = {}
    line = f.readline()
    line = line.rstrip("\n")
    if(line==""):
        break

    st_list = line.split("/")    # / 기준으로 split

    tmp_dic["분류"] = st_list[1]    # 1번 인덱스를 분류으로 정의
    tmp_dic["품목"] = st_list[2]    # 2번 인덱스를 품목으로 정의
    tmp_dic["가격"] = int(st_list[3])    # 3번 인덱스를 가격으로 정의 / 단 int로
    tmp_dic["재고"] = int(st_list[4])    # 4번 인덱스를 재고로 정의 / 단 int로

    goods[st_list[0]] = tmp_dic
    day_sale[st_list[0]] = 0

# menu 불러오기
while True:
    print("=" * 20)
    print("1. 결제 \n2. 물품관리 \n3. 매출관리 \n9. 종료")
    print("=" * 20, end="\n")
    select_num = input('실행할 번호를 입력해주세요.')

    if select_num == '1':    # 결제창
        tmp = pay.main(goods)
        update.main(goods, tmp, day_sale)

    elif select_num == '2':    # 물품관리창
        gmanagement.main(goods)

    elif select_num == '3':    # 매출관리창
        management.main(goods, day_sale)

    elif select_num == '9':    # 종료창
        ep.main(goods, day_sale)
        break

    else:    # 나머지는 오류로 처리
        print("다시 선택 하세요\n")

    print("\nSystem Down")