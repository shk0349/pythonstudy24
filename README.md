# pythonstudy24
Python AI 기초학습용

MBC 아카데미 컴퓨터 교육센터 수원점에서 AI기초 학습으로 Python 학습 진행용

https://wikidocs.net/book/1
<a href="https://github.com/devxb/gitanimals">
<img
  src="https://render.gitanimals.org/farms/shk0349"
  width="600"
  height="300"
/>
</a>

"""


    # 미션 / 커피 자판기를 리스트화 하여 for문으로 구현하라
    # 커피 종류가 5개 이상(커피명, 수량, 단가)
    # 사용자가 커피를 반복 구매 가능
    # 관리자가 판매 종료 후 통계를 볼 수 있도록 한다.
    # 단, 함수를 이용해서 만드시오.
    
    menu = []    # 메뉴
    price = []    # 가격
    su = []    # 수량(default)
    sell = []    # 판매수량(판매 시 1씩 증가 / default = 0)
    
    def print_auth():
        auth = input("""
        1. 관리자
        2. 고객
        3. 종료
        """)
        return auth
    
    def print_admin_menu():
        admin_menu = input("""
        1. 메뉴 추가
        2. 매출액 보기
        3. 처음으로
        """)
        return admin_menu
    
    def print_admin_menu_plus():
        menu.append(input("메뉴명 : "))
        price.append(int(input("가격 : ")))
        su.append(int(input("수량 : ")))
        sell.append(0)
        print("메뉴가 추가되었습니다.")
    
    def print_admin_total_sell():
        print("매출 결과")
        total_sell_price = 0
        for i in range(len(menu)):
            sell_price = price[i] * sell[i]
            total_sell_price += sell_price
            print("%s : %d잔" % (menu[i], sell[i]))
        print("매출액 : %d원" % total_sell_price)
    
    def print_error():
        print("잘못된 입력입니다.")
    
    def print_guest_menu():
        guest_menu = input("""
        1. 메뉴 보기
        2. 메뉴 주문
        3. 처음으로
        """)
        return guest_menu
    
    def print_menu():
        for i in range(len(menu)):
            print("%s : %d원" % (menu[i], price[i]))
            continue
    
    def print_select_menu():
        for i in range(len(menu)):
            print("%d. %s" % ((i+1), menu[i]))
        select_menu = input("메뉴 선택")
        return select_menu
    
    def main():
        auth = print_auth()
        while True:
            if auth == '1':
                admin_menu = print_admin_menu()
                if admin_menu == '1':
                    print_admin_menu_plus()
                elif admin_menu == '2':
                    print_admin_total_sell()
                elif admin_menu == '3':
                    auth = print_auth()
                else:
                    print_error
                    continue
            elif auth == '2':
                guest_menu = print_guest_menu()
                if guest_menu == '1':
                    print_menu()
                elif guest_menu == '2':
                    select_menu = print_select_menu()
                    if int(select_menu) <= len(menu):
                        if sell[int(select_menu) - 1] >= su[int(select_menu) - 1]:
                            print("재고가 부족합니다.")
                            continue
                        else:
                            pay = input("금액을 지불해주세요. %s 가격은 %d원 입니다." % (menu[int(select_menu) - 1], price[int(select_menu) - 1]))
                            if int(pay) > int(price[int(select_menu) - 1]):
                                sell[int(select_menu) - 1] = sell[int(select_menu) - 1] + 1
                                print("거스름돈은 %d원 입니다." % (int(pay) - price[int(select_menu) - 1]))
                            elif int(pay) == price[int(select_menu) - 1]:
                                sell[int(select_menu) - 1] = sell[int(select_menu) - 1] + 1
                                print("거스름돈은 없습니다.")
                            else:
                                print("금액이 부족합니다.")
                                continue
                    else:
                        print_error()
                        continue
                elif guest_menu == '3':
                    auth = print_auth()
                else:
                    print_error()
                    continue
            elif auth == '3':
                break
            else:
                print_error()
                auth = print_auth()
    
    main()


"""
